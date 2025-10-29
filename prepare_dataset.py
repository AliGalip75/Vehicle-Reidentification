import os
import pandas as pd

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
    '''
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
    '''

    # ---- Query ve Gallery ----
    make_csv(os.path.join(DATASET_ROOT, "image_query"), os.path.join(ANNOT_DIR, "query.csv"))
    make_csv(os.path.join(DATASET_ROOT, "image_test"), os.path.join(ANNOT_DIR, "gallery.csv"))


if __name__ == "__main__":
    main()
