import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ---- 1. Hazır model (ResNet50, ImageNet pretrained) ----
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
model.fc = nn.Identity()  # sınıflandırıcıyı kaldır, sadece feature al
model.eval()

# ---- 2. Görseli dönüştürme işlemleri ----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ---- 3. Özellik çıkarma fonksiyonu ----
def extract_feature(img_path):
    img = Image.open(img_path).convert('RGB')
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        feature = model(tensor)
    return feature.squeeze().numpy()

# ---- 4. Klasörleri belirt ----
query_dir = 'datasets/VeRi776/image_query'
gallery_dir = 'datasets/VeRi776/image_test'

# ---- 5. Query ve Gallery özelliklerini çıkar ----
query_features, gallery_features, gallery_paths = [], [], []

# Query klasöründen bir örnek alalım
query_path = os.path.join(query_dir, os.listdir(query_dir)[0])
query_feature = extract_feature(query_path)

# Gallery klasöründeki tüm resimlerden özellik çıkar
gallery_list = os.listdir(gallery_dir)[:1000]  # sadece ilk 1000 görsel
for img_name in gallery_list:
    img_path = os.path.join(gallery_dir, img_name)
    feat = extract_feature(img_path)
    gallery_features.append(feat)
    gallery_paths.append(img_path)

gallery_features = np.array(gallery_features)

# ---- 6. Benzerlik hesapla ----
similarities = cosine_similarity(query_feature.reshape(1, -1), gallery_features)[0]
top_indices = np.argsort(similarities)[::-1][:5]

print(f"\nQuery image: {query_path}")
print("\nTop 5 en benzer araç:")
for i in top_indices:
    print(f"{gallery_paths[i]} -> Benzerlik: {similarities[i]:.4f}")
