import argparse
import os
import sys
import math
import random
import numpy as np
import pandas as pd
import scipy.io
import torch
from torch.autograd import Variable
from torchvision import transforms
import matplotlib.pyplot as plt
import tqdm

# Bu dosyanın bulunduğu dizin ve üst dizini sys.path'e ekle
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from load_model import load_model_from_opts
from dataset import ImageDataset

# Tekrarlanabilirlik için deterministik algoritmalar
torch.use_deterministic_algorithms(True)

##########################
# Seçenekler / Argümanlar
# ------------------------

parser = argparse.ArgumentParser(
    description="Bir re-id modeli için örnek sorgular ve getirilen galeri görüntülerini gösterir."
)
#--------------------------
parser.add_argument(
    "--model_opts",
    required=True,
    type=str,
    help="Kullanılacak model ayarları (use_saved_mat varsa kullanılmaz)."
)
parser.add_argument(
    "--checkpoint",
    required=True, 
    type=str,
    help="Model ağırlık dosyası (checkpoint)."
)
#--------------------------
parser.add_argument(
    "--query_csv_path", 
    default="../../datasets/id_split_cityflow_query.csv",
    type=str, 
    help="Sorgu (query) görsellerine ait CSV yolu."
)
parser.add_argument(
    "--gallery_csv_path", 
    default="../../datasets/id_split_cityflow_gallery.csv",
    type=str, 
    help="Galeri görsellerine ait CSV yolu."
)
parser.add_argument(
    "--data_dir", 
    type=str, 
    default="../../datasets/",
    help="Görsel veri kümeleri için kök dizin."
)
#--------------------------
parser.add_argument(
    "--input_size", 
    type=int, 
    default=224,
    help="Model için giriş görüntü boyutu (h=w)."
)
parser.add_argument(
    "--batchsize", 
    type=int, 
    default=64, 
    help="Özellik çıkarımı sırasında kullanılacak batch boyutu (daha büyük = daha hızlı, daha çok bellek)."
)
#--------------------------
parser.add_argument(
    "--num_images", 
    type=int, 
    default=29,
    help="Gösterilecek galeri görüntüsü sayısı."
)
parser.add_argument(
    "--imgs_per_row", 
    type=int, 
    default=6,
    help="Görselleştirmede satır başına düşen görüntü sayısı."
)
#--------------------------
parser.add_argument(
    "--use_saved_mat", 
    action="store_true",
    help="Önceden test.py ile hesaplanmış 'pytorch_result.mat' özelliklerini kullan."
)
parser.add_argument(
    "--no_cams", 
    action="store_true",
    help="Kamera filtresini devre dışı bırak."
)

args = parser.parse_args()

# Cihaz seçimi
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

h, w = args.input_size, args.input_size

######################################################################
# Veriyi Yükle
# ------------

# Test/eval dönüşümleri (ImageNet normları)
data_transforms = transforms.Compose([
    transforms.Resize((h, w), interpolation=3),  # bicubic
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# CSV'leri oku
query_df = pd.read_csv(args.query_csv_path)
gallery_df = pd.read_csv(args.gallery_csv_path)

# Ortak sınıf listesi (id kolonlarından)
classes = sorted(list(pd.concat([query_df["id"], gallery_df["id"]]).unique()))

# Kamera filtresi kullanılacak mı? (her iki CSV'de 'cam' kolonu olmalı ve --no_cams verilmemeli)
use_cam = ('cam' in query_df.columns and 'cam' in gallery_df.columns) and (not args.no_cams)

# PyTorch Dataset sarmalayıcıları
image_datasets = {
    "query": ImageDataset(args.data_dir, query_df, "id", classes, transform=data_transforms),
    "gallery": ImageDataset(args.data_dir, gallery_df, "id", classes, transform=data_transforms),
}

# DataLoader'lar (sıralama bozulmaması için shuffle=False)
dataloaders = {
    x: torch.utils.data.DataLoader(
        image_datasets[x],
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=2
    )
    for x in ['gallery', 'query']
}

######################################################################
# Özellik Çıkarımı (Feature Extraction)
# ------------------------------------
# Eğitilmiş modelden özellik vektörleri üretir.

def fliplr(img: torch.Tensor) -> torch.Tensor:
    """Görüntüyü yatay eksende çevirir (N x C x H x W)."""
    inv_idx = torch.arange(img.size(3) - 1, -1, -1, device=img.device).long()
    return img.index_select(3, inv_idx)

def extract_features(model: torch.nn.Module, dataloader):
    """
    Bir DataLoader içindeki tüm görüntülerden özellik çıkarır.
    Test-time augmentation olarak yatay çevirme (flip) uygular ve birleştirir.
    Çıkışlar L2-normalize edilir.
    """
    img_count = 0

    # Çıktı boyutunu belirlemek için bir dummy ileri geçiş
    dummy = next(iter(dataloader))[0].to(device)
    with torch.no_grad():
        output = model(dummy)
    feature_dim = output.shape[1]

    labels = []

    for idx, data in enumerate(tqdm.tqdm(dataloader)):
        X, y = data
        n, c, h, w = X.size()
        img_count += n

        # Bu batch için özellik biriktirici (GPU'da)
        ff = torch.zeros(n, feature_dim, dtype=torch.float32, device=device)

        # Etiketleri listeye ekle (sıra korunur)
        for lab in y:
            labels.append(lab)

        # Orijinal + yatay çevrilmiş ileri geçişleri topla
        with torch.no_grad():
            for i in range(2):
                X_use = fliplr(X) if i == 1 else X
                input_X = Variable(X_use.to(device))
                outputs = model(input_X)
                ff += outputs

        # L2 normalizasyonu
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))

        # Tüm veri kümesi için CPU tarafında ön-tanımlı tensör oluştur ve doldur
        if idx == 0:
            features = torch.zeros(len(dataloader.dataset), ff.shape[1], dtype=torch.float32)

        start = idx * args.batchsize
        end = min((idx + 1) * args.batchsize, len(dataloader.dataset))
        # ---- DÜZELTME: GPU tensörünü CPU'ya taşıyarak kopyala ----
        features[start:end, :] = ff.detach().cpu()

    return features, labels

def extract_feature(model: torch.nn.Module, img: torch.Tensor) -> torch.Tensor:
    """
    Tek bir görüntüden (veya 4D batch'ten) L2-normalize edilmiş özellik çıkarır.
    Orijinal + yatay çevrilmiş ileri geçişlerin toplamını kullanır.
    """
    with torch.no_grad():
        if len(img.shape) == 3:
            img = torch.unsqueeze(img, 0)
        img = img.to(device)

        feature = model(img).reshape(-1)
        flipped_feature = model(fliplr(img)).reshape(-1)
        feature = feature + flipped_feature

        fnorm = torch.norm(feature, p=2)
        feature = feature.div(fnorm)
    return feature

def get_scores(query_feature: torch.Tensor, gallery_features: torch.Tensor) -> np.ndarray:
    """
    Kosinüs benzerliği ile (galeri_features @ query_feature) skoru hesaplar.
    """
    query = query_feature.view(-1, 1)           # (D, 1)
    score = torch.mm(gallery_features, query)   # (N, 1)
    score = score.squeeze(1).cpu().numpy()      # (N,)
    return score

def show_query_result(axes, query_img, gallery_imgs, query_label, gallery_labels, query_id, gallery_ids):
    """
    Sorgu ve en iyi eşleşen galeri görüntülerini plot eder.
    - Sorgu görüntüsü siyah çerçeveli, galeri görüntüleri:
      * ID tam eşleşirse: yeşil çerçeve
      * Eşleşmezse: kırmızı çerçeve
    Üstte metin olarak ID gösterilir.
    """
    query_trans = transforms.Pad(4, 0)
    good_trans = transforms.Pad(4, (0, 255, 0))
    bad_trans = transforms.Pad(4, (255, 0, 0))

    for idx, img in enumerate([query_img] + gallery_imgs):
        img = img.resize((128, 128))
        if idx == 0:
            # Sorgu görseli
            img = query_trans(img)
            label = query_id
        else:
            # Galeri görseli: ID eşleşmesine göre kenarlık
            is_good = (query_id == gallery_ids[idx - 1])
            img = good_trans(img) if is_good else bad_trans(img)
            label = gallery_ids[idx - 1]

        ax = axes.flat[idx]
        ax.imshow(img)
        # ID'yi üst banda yaz
        ax.text(
            0.5, 0.95, f"ID: {label}", transform=ax.transAxes,
            color='white', fontsize=8, ha='center', va='top',
            bbox=dict(facecolor='black', alpha=0.5)
        )

    # Tüm eksenleri gizle
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_xticks([], minor=True)
        ax.set_yticks([])
        ax.set_yticks([], minor=True)
        ax.axis("off")

######################################################################
# Sorguları Çalıştır
# ------------------

if args.use_saved_mat:
    # Önceden kaydedilmiş .mat dosyasından özellikleri yükle
    saved_res = scipy.io.loadmat("pytorch_result.mat")
    gallery_features = torch.Tensor(saved_res["gallery_f"])
    gallery_labels = saved_res["gallery_label"].reshape(-1)
    query_features = torch.Tensor(saved_res["query_f"])
    query_labels = saved_res["query_label"].reshape(-1)
else:
    # Modeli yükle ve değerlendirme moduna al
    model = load_model_from_opts(args.model_opts, args.checkpoint, remove_classifier=True)
    model.eval()
    model.to(device)

    print("Galeri özellikleri hesaplanıyor ...")
    with torch.no_grad():
        gallery_features, gallery_labels = extract_features(model, dataloaders["gallery"])
        gallery_labels = np.array(gallery_labels)

# Sorgu kümesi ve rastgele sorgu sırası
dataset = image_datasets["query"]
queries = list(range(len(dataset)))
random.shuffle(queries)

def on_key(event):
    """
    Klavye kısayolları:
      ← : bir önceki sorgu
      → : bir sonraki sorgu
      Enter : Son plot'u PDF olarak kaydet (reid_query_result.pdf)
    """
    global curr_idx
    if event.key == "left":
        curr_idx = (curr_idx - 1) if curr_idx > 0 else len(queries) - 1
    elif event.key == "right":
        curr_idx = (curr_idx + 1) if curr_idx < len(queries) - 1 else 0
    elif event.key == "enter":
        fig.savefig("reid_query_result.pdf", pad_inches=0, bbox_inches='tight')
    else:
        return
    refresh_plot()

def refresh_plot():
    """
    Mevcut sorgunun sonucunu hesaplar ve tuvale çizer.
    Kamera filtresi açıksa, aynı kameradan gelen galeri görüntülerini çıkarır.
    """
    query_index = queries[curr_idx]

    # Sorgu özelliği ve (debug) etiket bilgisi
    if args.use_saved_mat:
        q_feature = query_features[query_index]
        y = query_labels[query_index]
    else:
        X, y = dataset[query_index]
        with torch.no_grad():
            q_feature = extract_feature(model, X).cpu()  # CPU'da skorlayacağız

    # Kamera filtresi: aynı kameradan gelen galeri örneklerini çıkar
    if use_cam:
        curr_cam = query_df["cam"].iloc[query_index]
        good_gallery_idx = (gallery_df["cam"] != curr_cam).values  # bool numpy maske
        gallery_orig_idx = np.where(good_gallery_idx)[0]
        gal_features = gallery_features[good_gallery_idx]
    else:
        gallery_orig_idx = np.arange(len(gallery_df))
        gal_features = gallery_features

    # Skorla ve en iyi eşleşmeleri sırala (azalan)
    gallery_scores = get_scores(q_feature, gal_features)
    idx = np.argsort(gallery_scores)[::-1]

    # Etiket/ID listeleri (sıralamaya göre)
    if use_cam:
        g_labels = gallery_labels[gallery_orig_idx][idx]
        g_ids = gallery_df["id"].iloc[gallery_orig_idx[idx]].values
    else:
        g_labels = gallery_labels[idx]
        g_ids = gallery_df["id"].iloc[idx].values

    # Konsola kısa özet yazdır
    print(f"Query Index: {query_index}, Query ID: {query_df['id'].iloc[query_index]}, Query Label: {y}")
    print(f"Top 5 Gallery IDs: {g_ids[:5]}")
    print(f"Top 5 Gallery Labels: {g_labels[:5]}")
    print(f"Top 5 Gallery Scores: {gallery_scores[idx][:5]}")

    # Görselleri getir ve plot et
    q_img = dataset.get_image(query_index)
    q_id = query_df["id"].iloc[query_index]
    g_imgs = [image_datasets["gallery"].get_image(gallery_orig_idx[i]) for i in idx[:args.num_images]]
    show_query_result(axes, q_img, g_imgs, y, g_labels, q_id, g_ids)
    fig.canvas.draw()
    fig.canvas.flush_events()

# Tuval ve eksenleri hazırla
n_rows = math.ceil((1 + args.num_images) / args.imgs_per_row)
fig, axes = plt.subplots(n_rows, args.imgs_per_row, figsize=(12, 15))
fig.canvas.mpl_connect('key_press_event', on_key)

HELP_TXT = "← ve → ile sorgular arasında gez. Enter ile PDF olarak kaydet (reid_query_result.pdf)."
print(HELP_TXT)

curr_idx = 0
refresh_plot()
plt.show()
