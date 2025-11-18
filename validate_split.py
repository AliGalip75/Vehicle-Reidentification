# validate_splits.py
import os
import sys
import hashlib
import pandas as pd

# >>> KULLANIM <<<
# python validate_splits.py \
#   --data_dir datasets \
#   --train datasets/annot/train_annot.csv \
#   --val   datasets/annot/val_annot.csv \
#   --query datasets/annot/query.csv \
#   --gallery datasets/annot/gallery.csv \
#   [--strict_idcam] [--hash_check]

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--data_dir", required=True, help="CSV 'path' sütununa baz alınacak kök (örn. datasets)")
ap.add_argument("--train", required=True)
ap.add_argument("--val", required=True)
ap.add_argument("--query", required=True)
ap.add_argument("--gallery", required=True)
ap.add_argument("--strict_idcam", action="store_true",
                help="train & val arasında aynı (id,cam) varsa uyar (katı mod)")
ap.add_argument("--hash_check", action="store_true",
                help="splitler arası aynı içerik var mı (md5) bakar. Yavaş olabilir.")
args = ap.parse_args()

def load_csv(p):
    df = pd.read_csv(p)
    need = {"path","id"}
    assert need.issubset(df.columns), f"{p} zorunlu sütunları içermiyor: {need}"
    # id tipi
    try:
        df["id"] = df["id"].astype(int)
    except Exception:
        raise AssertionError(f"{p} -> 'id' sütunu tamsayıya çevrilemedi.")
    # cam varsa int yap
    if "cam" in df.columns:
        # cam bazen float/str olabilir; güvenli çeviri:
        df["cam"] = pd.to_numeric(df["cam"], errors="coerce").astype("Int64")
    return df

def exists_check(df, data_dir):
    # dosya var mı?
    paths = df["path"].tolist()
    joined = [os.path.join(data_dir, p) for p in paths]
    mask = [os.path.exists(p) for p in joined]
    missing = df.loc[[not m for m in mask], "path"].head(20)
    if len(missing) > 0:
        print(f"❌ Dosya bulunamadı (ilk 20):\n{missing.to_string(index=False)}")
    else:
        print("✅ Tüm dosyalar mevcut.")

def dup_check(df, name):
    ok = True
    d1 = df[df.duplicated(subset=["path"], keep=False)]
    if len(d1) > 0:
        ok = False
        print(f"❌ {name}: aynı path birden çok kez var (ilk 10):")
        print(d1.head(10)[["path","id"]].to_string(index=False))
    d2 = df[df.duplicated(subset=["path","id"], keep=False)]
    if len(d2) > 0:
        ok = False
        print(f"⚠️  {name}: aynı (path,id) tekrarları (ilk 10):")
        print(d2.head(10)[["path","id"]].to_string(index=False))
    if ok:
        print(f"✅ {name}: tekillik OK (path).")

def overlap_paths(a, b, name_a, name_b):
    s = set(a["path"])
    t = set(b["path"])
    inter = sorted(list(s & t))
    if inter:
        print(f"❌ {name_a} ↔ {name_b}: aynı görseller var (ilk 20):")
        for p in inter[:20]:
            print("   ", p)
    else:
        print(f"✅ {name_a} ↔ {name_b}: path çatışması yok.")

def id_intersection(train, test_all):
    inter = sorted(list(set(train["id"]) & set(test_all["id"])))
    if inter:
        print(f"❌ train ID'leri test (query+gallery) ile kesişiyor. Örnek ilk 10 id: {inter[:10]}")
    else:
        print("✅ train ID'leri test ile ayrık (disjoint).")

def strict_idcam_check(train, val):
    if "cam" not in train.columns or "cam" not in val.columns:
        print("ℹ️  strict_idcam: cam sütunu yok, atlanıyor.")
        return
    pairs_train = set(zip(train["id"].tolist(), train["cam"].tolist()))
    pairs_val   = set(zip(val["id"].tolist(),   val["cam"].tolist()))
    inter = pairs_train & pairs_val
    if inter:
        print(f"⚠️  train ↔ val aynı (id,cam) çiftleri mevcut (ilk 20):")
        for x in list(inter)[:20]:
            print("   ", x)
        print("    Not: Bu tam aynı kare demek değildir ama kaçak riski. En azından aynı path’in kesişmediğinden emin ol (aşağıdaki kontrol zaten bakıyor).")
    else:
        print("✅ train ↔ val (id,cam) ayrık (katı kontrol).")

def query_has_match(query, gallery):
    # her query için galeride en az bir aynı id (tercihen cam !=) var mı
    g_by_id = gallery.groupby("id")
    bad = []
    for i, row in query.iterrows():
        qid = row["id"]
        if qid not in g_by_id.groups:
            bad.append((row["path"], qid, row.get("cam", pd.NA)))
        else:
            if "cam" in query.columns and "cam" in gallery.columns and pd.notna(row.get("cam", pd.NA)):
                cand = g_by_id.get_group(qid)
                if (cand["cam"] != row["cam"]).sum() == 0:
                    # aynı id var ama hepsi aynı cam -> protokole göre junk olabilir
                    bad.append((row["path"], qid, row["cam"]))
    if bad:
        print(f"⚠️  Bazı query'lerin galeride uygun eşleşmesi yok (veya cam aynı). İlk 20:")
        for x in bad[:20]:
            print("   path:", x[0], " id:", x[1], " cam:", x[2])
    else:
        print("✅ Tüm query’ler için galeride en az bir uygun eşleşme var.")

def md5_path(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()

def hash_overlap(a, b, name_a, name_b, data_dir, limit=2000):
    # büyük setlerde sınırlayalım
    import itertools
    def compute_df_hashes(df):
        rows = []
        for p in df["path"].head(limit):
            full = os.path.join(data_dir, p)
            if os.path.exists(full):
                rows.append((p, md5_path(full)))
        return pd.DataFrame(rows, columns=["path","md5"])
    ha = compute_df_hashes(a)
    hb = compute_df_hashes(b)
    inter = pd.merge(ha, hb, on="md5", suffixes=(f"_{name_a}", f"_{name_b}"))
    if len(inter) > 0:
        print(f"❌ {name_a} ↔ {name_b}: aynı içerik (md5) bulundu (ilk 10):")
        print(inter.head(10).to_string(index=False))
    else:
        print(f"✅ {name_a} ↔ {name_b}: içerik bazlı çakışma yok (örneklem {limit}).")

def main():
    train  = load_csv(args.train)
    val    = load_csv(args.val)
    query  = load_csv(args.query)
    gallery= load_csv(args.gallery)

    print("== Dosya varlık kontrolü ==")
    exists_check(train, args.data_dir)
    exists_check(val, args.data_dir)
    exists_check(query, args.data_dir)
    exists_check(gallery, args.data_dir)

    print("\n== CSV içi tekillik ==")
    dup_check(train, "train")
    dup_check(val, "val")
    dup_check(query, "query")
    dup_check(gallery, "gallery")

    print("\n== Splitler arası path çakışması ==")
    overlap_paths(train, val, "train", "val")
    overlap_paths(query, gallery, "query", "gallery")
    overlap_paths(train, query, "train", "query")
    overlap_paths(train, gallery, "train", "gallery")
    overlap_paths(val, query, "val", "query")
    overlap_paths(val, gallery, "val", "gallery")

    print("\n== ID ayrıklığı (train vs test) ==")
    test_all = pd.concat([query[["id"]], gallery[["id"]]], axis=0).drop_duplicates()
    id_intersection(train, test_all)

    if args.strict_idcam:
        print("\n== Katı kontrol: (id,cam) train ↔ val ==")
        strict_idcam_check(train, val)

    print("\n== Query eşleşme uygunluğu ==")
    query_has_match(query, gallery)

    if args.hash_check:
        print("\n== İçerik (md5) çakışması (örneklem) ==")
        hash_overlap(train, val, "train", "val", args.data_dir)
        hash_overlap(query, gallery, "query", "gallery", args.data_dir)
        hash_overlap(train, query, "train", "query", args.data_dir)
        hash_overlap(train, gallery, "train", "gallery", args.data_dir)

    print("\n✔︎ Kontrol tamam.")

if __name__ == "__main__":
    main()
