import xml.etree.ElementTree as ET
import pandas as pd
import os

# VeRi kök dizini (etiket XML'leri ve görüntü klasörleri bu dizinin altında)
root = r"VeRi"

def xml_to_csv(xml_path, img_folder):
    """
    Verilen XML dosyasındaki <Item> etiketlerini okuyup
    resim yolu ve araç kimliğini (vehicleID) CSV'ye dönüştürür.

    Parametreler:
        xml_path (str): Etiketlerin bulunduğu XML dosyasının yolu.
        img_folder (str): Görüntülerin yer aldığı klasör adı (root altında).

    Dönüş:
        pd.DataFrame: 'path' ve 'id' sütunlarını içeren tablo.
    """
    # XML'i gb2312 ile okumayı dene, başarısız olursa UTF-8'e düş
    try:
        with open(xml_path, 'r', encoding='gb2312') as f:
            xml_data = f.read()
    except UnicodeError:
        with open(xml_path, 'r', encoding='utf-8') as f:
            xml_data = f.read()

    # XML'i parse et
    try:
        root_xml = ET.fromstring(xml_data)
    except ET.ParseError as e:
        raise RuntimeError(f"XML parse hatası: {xml_path} -> {e}")

    data = []

    # Tüm <Item> etiketlerini dolaş ve gerekli alanları topla
    for item in root_xml.findall('.//Item'):
        img_name = item.get('imageName')
        vid = item.get('vehicleID')

        # imageName ve vehicleID yoksa atla
        if not img_name or not vid:
            continue

        # Tam görüntü yolunu oluştur (OS bağımsız)
        path = os.path.join(img_folder, img_name)

        # vehicleID'yi tamsayıya çevir (geçersizse atla)
        try:
            vid_int = int(vid)
        except ValueError:
            continue

        data.append([path, vid_int])

    print(f"{xml_path} içinde {len(data)} satır bulundu.")
    return pd.DataFrame(data, columns=['path', 'id'])


# Eğitim ve test XML'lerini DataFrame'e dönüştür
train_df = xml_to_csv(os.path.join(root, "train_label.xml"), "image_train")
test_df  = xml_to_csv(os.path.join(root, "test_label.xml"),  "image_test")

# Çıkış klasörünü oluştur (varsa hata vermez)
os.makedirs(os.path.join(root, "annot"), exist_ok=True)

# CSV'leri kaydet
train_csv_path = os.path.join(root, "annot", "train_annot.csv")
val_csv_path   = os.path.join(root, "annot", "val_annot.csv")

# Not: path sütunları göreli olduğundan, veri yüklerken kök olarak 'root' kullanmayı unutmayın.
train_df.to_csv(train_csv_path, index=False)
test_df.to_csv(val_csv_path, index=False)

print("✅ CSV dosyaları başarıyla oluşturuldu!")
print(f"   - Eğitim: {train_csv_path}")
print(f"   - Doğrulama: {val_csv_path}")
