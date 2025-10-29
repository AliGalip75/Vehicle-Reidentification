import argparse
import os
import torch
from torchvision import transforms
import pandas as pd
from PIL import Image
import numpy as np
import torch.nn.functional as F

# Model ve veri seti için gerekli modüller
from load_model import load_model_from_opts
from dataset import ImageDataset

# Argümanlar
parser = argparse.ArgumentParser(description="Manuel görüntü testi")
parser.add_argument("--model_opts", required=True, type=str, help="Modelin opts.yaml dosyası")
parser.add_argument("--checkpoint", required=True, type=str, help="Eğitilmiş modelin .pth dosyası")
parser.add_argument("--query_image", required=True, type=str, help="Test edilecek sorgu görüntüsünün yolu")
parser.add_argument("--gallery_csv_path", required=True, type=str, help="Galeri CSV dosyası")
parser.add_argument("--data_dir", required=True, type=str, help="Veri seti kök dizini")
parser.add_argument("--top_k", type=int, default=5, help="Gösterilecek en iyi eşleşme sayısı")
args = parser.parse_args()

# Cihaz ayarları
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Veri dönüşümleri (test.py ile aynı olmalı)
h, w = 224, 224
data_transforms = transforms.Compose([
    transforms.Resize((h, w), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Modeli yükle
model = load_model_from_opts(args.model_opts, ckpt=args.checkpoint, remove_classifier=True)
model.eval()
model.to(device)

# Galeri veri setini yükle
gallery_df = pd.read_csv(args.gallery_csv_path)
classes = sorted(list(gallery_df["id"].unique()))  # ID'leri sıralı al
gallery_dataset = ImageDataset(args.data_dir, gallery_df, "id", classes, transform=data_transforms)
gallery_loader = torch.utils.data.DataLoader(gallery_dataset, batch_size=32, shuffle=False, num_workers=2)

# Özellik çıkarma fonksiyonu (test.py'den uyarlandı)
def fliplr(img):
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long().to(img.device)
    return img.index_select(3, inv_idx)

def extract_feature(model, img):
    img = img.to(device)
    feature = model(img).reshape(-1)
    img = fliplr(img)
    flipped_feature = model(img).reshape(-1)
    feature += flipped_feature
    fnorm = torch.norm(feature, p=2)
    return feature.div(fnorm)

def extract_gallery_features(model, dataloader):
    features = []
    labels = []
    for data in dataloader:
        X, y = data
        X = X.to(device)
        with torch.no_grad():
            ff = model(X)
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        features.append(ff.cpu())
        labels.extend(y.numpy())
    return torch.cat(features), np.array(labels)

# Galeri özelliklerini çıkar
print("Galeri özellikleri çıkarılıyor...")
gallery_features, gallery_labels = extract_gallery_features(model, gallery_loader)

# Sorgu görüntüsünü yükle
query_img = Image.open(args.query_image).convert("RGB")
query_img = data_transforms(query_img)
query_img = query_img.unsqueeze(0)  # Batch boyutu ekle
with torch.no_grad():
    query_feature = extract_feature(model, query_img)

# Eşleşmeleri hesapla
query_feature = query_feature.view(-1, 1)
scores = torch.mm(gallery_features, query_feature).squeeze(1).cpu().numpy()
indices = np.argsort(scores)[::-1][:args.top_k]  # En yüksek skorlu top_k indeksi

# Sonuçları yazdır
print(f"\nSorgu görüntüsü: {args.query_image}")
print(f"En iyi {args.top_k} eşleşme:")
for i, idx in enumerate(indices):
    score = scores[idx]
    label = gallery_labels[idx]
    image_path = gallery_dataset.image_paths[idx]
    print(f"Rank {i+1}: Skor={score:.4f}, Etiket={label}, Görüntü={image_path}")