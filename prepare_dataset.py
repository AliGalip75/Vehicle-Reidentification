import os
import pandas as pd
'''
DATASET_ROOT = "VeRi"

# Annotasyon dosyalarının kaydedileceği klasör
ANNOT_DIR = os.path.join("datasets", "annot")
os.makedirs(ANNOT_DIR, exist_ok=True)


def make_csv(img_dir, out_csv):
    data = []
    for fname in os.listdir(img_dir):
        if fname.endswith(".jpg"):
            parts = fname.split('_')
            vid = int(parts[0])                     # araç ID
            cam = int(parts[1][1:])                 # 'c001' → 1
            rel_path = os.path.join(os.path.basename(img_dir), fname)
            data.append([rel_path, vid, cam])
    df = pd.DataFrame(data, columns=["path", "id", "cam"])
    df.to_csv(out_csv, index=False)
    print(f"[OK] {out_csv} kaydedildi. Toplam {len(df)} resim.")


def main():
    
    # ---- Train + Validation split ----
    all_train = []
    train_dir = os.path.join(DATASET_ROOT, "image_train")
    for fname in os.listdir(train_dir):
        if fname.endswith(".jpg"):
            vid = int(fname.split('_')[0])
            rel_path = os.path.join("image_train", fname)
            all_train.append([rel_path, vid])

    df = pd.DataFrame(all_train, columns=["path", "id"])
    train_split = int(len(df) * 0.9)  # %90 train, %10 val
    df.iloc[:train_split].to_csv(os.path.join(ANNOT_DIR, "train_annot.csv"), index=False)
    df.iloc[train_split:].to_csv(os.path.join(ANNOT_DIR, "val_annot.csv"), index=False)
    print(f"[OK] train_annot.csv ve val_annot.csv kaydedildi.")
    

    # ---- Query ve Gallery ----
    make_csv(os.path.join(DATASET_ROOT, "image_query"), os.path.join(ANNOT_DIR, "query.csv"))
    make_csv(os.path.join(DATASET_ROOT, "image_test"), os.path.join(ANNOT_DIR, "gallery.csv"))

'''
import os
import re
import pandas as pd
import xml.etree.ElementTree as ET

ROOT = "datasets"                         
VERI = os.path.join(ROOT, "VeRi")
ANNOT = os.path.join(ROOT, "annot")
os.makedirs(ANNOT, exist_ok=True)

def read_txt_list(pth):
    """Her satırı bir öğe olan düz liste oku (boş satırları atlar)."""
    out = []
    with open(pth, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(line)
    return out

def read_kv_list(pth):
    """
    list_color.txt / list_type.txt formatını sözlüğe çevir.
    Satır örnekleri genelde:  "1 black"  ya da  "1,black"
    """
    mp = {}
    with open(pth, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # sayıyı ve ismi yakala (virgül/boşluk toleransı)
            m = re.match(r"^\s*(\d+)\s*[, ]\s*(.+?)\s*$", line)
            if m:
                k = int(m.group(1))
                v = m.group(2).strip()
                mp[k] = v
            else:
                # "1 black sedan" gibi çok boşluklu olabilir
                parts = re.split(r"[,\s]+", line)
                if parts and parts[0].isdigit():
                    k = int(parts[0])
                    v = " ".join(parts[1:]).strip() if len(parts) > 1 else str(k)
                    mp[k] = v
    return mp

def _to_int_or_none(val):
    """'c001', '001', '1' gibi stringlerden son sayısal kısmı çekip int'e çevirir; yoksa None."""
    if val is None:
        return None
    s = str(val).strip()
    m = re.search(r"(\d+)$", s)  # sonda geçen rakamları yakala
    return int(m.group(1)) if m else None

def parse_xml_labels(xml_path):
    """
    train_label.xml / test_label.xml okur.
    Beklenen attribute'lar:
      imageName, vehicleID, cameraID, colorID, typeID
    - 'cameraID' 'c001' gibi olabilir -> 1'e çevrilir.
    - color/type yoksa None bırakılır.
    """
    # VeRi XML bazen gb2312 ile kaydedilmiş; UTF-8'e düşerek oku
    try:
        with open(xml_path, "r", encoding="gb2312") as f:
            xml_data = f.read()
    except UnicodeError:
        with open(xml_path, "r", encoding="utf-8", errors="ignore") as f:
            xml_data = f.read()

    root = ET.fromstring(xml_data)

    rows = []
    for item in root.findall(".//Item"):
        img_name = item.get("imageName")
        vid      = _to_int_or_none(item.get("vehicleID"))
        cam      = _to_int_or_none(item.get("cameraID"))
        color    = _to_int_or_none(item.get("colorID"))
        vtype    = _to_int_or_none(item.get("typeID"))

        # imageName, id ve cam zorunlu; biri yoksa atla
        if not img_name or vid is None or cam is None:
            continue

        rows.append({
            "imageName": img_name,
            "id":        vid,
            "cam_id":    cam,
            "color_id":  color,
            "type_id":   vtype,
        })

    return pd.DataFrame(rows)

def add_pretty_attrs(df, color_map, type_map):
    """color_id/type_id → color/type isimlerini ekle."""
    df["color"] = df["color_id"].map(color_map) if color_map else None
    df["type"]  = df["type_id"].map(type_map)   if type_map  else None
    return df

def make_split_from_xml(xml_df, subdir, keep_names=None):
    """
    XML’den gelen tabloyu CSV biçimine çevir:
      path,id,cam,color,type
    - subdir: 'image_train', 'image_test' (veya 'image_gallery'), 'image_query'
    - keep_names: sadece bu dosyalar (name_*.txt) tutulur; None ise hepsi.
    """
    # image_test vs image_gallery otomatik tespit
    if subdir == "image_test" and not os.path.isdir(os.path.join(VERI, "image_test")):
        subdir = "image_gallery"
    if subdir == "image_gallery" and not os.path.isdir(os.path.join(VERI, "image_gallery")):
        subdir = "image_test"

    df = xml_df.copy()
    if keep_names is not None:
        keep = set(keep_names)
        df = df[df["imageName"].isin(keep)]

    # path: --data_dir=datasets olduğundan VeRi/... ile başlayacak şekilde
    df["path"] = df["imageName"].apply(lambda x: os.path.join("VeRi", subdir, x))
    df = df.rename(columns={"id": "id", "cam_id": "cam"})

    # Sadece gerekli sütunlar
    cols = ["path", "id", "cam", "color", "type"]
    # Eksik sütunlar varsa ekle
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df[cols].reset_index(drop=True)

def main():
    # 1) Renk/Tip sözlükleri
    color_map = {}
    type_map  = {}
    color_list = os.path.join(VERI, "list_color.txt")
    type_list  = os.path.join(VERI, "list_type.txt")
    if os.path.exists(color_list):
        color_map = read_kv_list(color_list)
    if os.path.exists(type_list):
        type_map  = read_kv_list(type_list)

    # 2) XML’leri parse et
    train_xml = os.path.join(VERI, "train_label.xml")
    test_xml  = os.path.join(VERI, "test_label.xml")
    train_df0 = parse_xml_labels(train_xml)
    test_df0  = parse_xml_labels(test_xml)

    # 3) İsimlere göre filtre listeleri (resmî split)
    q_list = read_txt_list(os.path.join(VERI, "name_query.txt"))
    g_list = read_txt_list(os.path.join(VERI, "name_test.txt"))

    # 4) Renk/tip isimlerini ekle
    train_df0 = add_pretty_attrs(train_df0, color_map, type_map)
    test_df0  = add_pretty_attrs(test_df0,  color_map, type_map)

    # 5) Çıkış tablolarını hazırla
    train_csv  = make_split_from_xml(train_df0, "image_train")
    gallery_csv= make_split_from_xml(test_df0,  "image_test",  keep_names=g_list)
    query_csv  = make_split_from_xml(test_df0,  "image_query", keep_names=q_list)

    # 6) Kaydet
    train_csv.to_csv (os.path.join(ANNOT, "train_full.csv"),   index=False)
    gallery_csv.to_csv(os.path.join(ANNOT, "gallery_full.csv"), index=False)
    query_csv.to_csv (os.path.join(ANNOT, "query_full.csv"),    index=False)

    print("✅ Yazıldı:")
    print("  - datasets/annot/train_full.csv")
    print("  - datasets/annot/gallery_full.csv")
    print("  - datasets/annot/query_full.csv")

if __name__ == "__main__":
    main()

