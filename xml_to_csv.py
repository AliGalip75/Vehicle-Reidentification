import xml.etree.ElementTree as ET
import pandas as pd
import os

root = r"VeRi"

def xml_to_csv(xml_path, img_folder):
    # gb2312 encoding ile aç
    with open(xml_path, 'r', encoding='gb2312') as f:
        xml_data = f.read()

    root_xml = ET.fromstring(xml_data)
    data = []

    # Item etiketlerini oku
    for item in root_xml.findall('.//Item'):
        img_name = item.get('imageName')
        vid = item.get('vehicleID')
        if img_name and vid:
            path = os.path.join(img_folder, img_name)
            data.append([path, int(vid)])

    print(f"{xml_path} içinde {len(data)} satır bulundu.")
    return pd.DataFrame(data, columns=['path', 'id'])


train_df = xml_to_csv(f"{root}/train_label.xml", "image_train")
test_df  = xml_to_csv(f"{root}/test_label.xml", "image_test")

os.makedirs(f"{root}/annot", exist_ok=True)
train_df.to_csv(f"{root}/annot/train_annot.csv", index=False)
test_df.to_csv(f"{root}/annot/val_annot.csv", index=False)

print("✅ CSV dosyaları başarıyla oluşturuldu!")
