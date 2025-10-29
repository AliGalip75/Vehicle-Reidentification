import argparse  # Komut satırı argümanlarını parse etmek için
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
import tqdm  # progress bar

# SCRIPT_DIR: script'in bulunduğu klasörün yolunu al
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# parent dizini Python path'ine ekle (load_model ve dataset modülleri orada olabilir)
sys.path.append(os.path.dirname(SCRIPT_DIR))

# Projendeki yardımcı modüller
from load_model import load_model_from_opts
from dataset import ImageDataset

######################################################################
# Options (komut satırı argümanları)
######################################################################

parser = argparse.ArgumentParser(
    description="Show sample queries and retrieved gallery images for a reid model")
# --model_opts: eğitim sırasında kullanılan model seçeneklerini içeren YAML vs
parser.add_argument("--model_opts", required=True,
                    type=str, help="model to use, if --use_saved_mat is provided then this is not used.")
# --checkpoint: model ağırlıkları (net_X.pth gibi)
parser.add_argument("--checkpoint", required=True,
                    type=str, help="checkpoint to load for model.")
# query ve gallery CSV'lerinin yolları (her satırda path,id ve opsiyonel cam gibi sütunlar olmalı)
parser.add_argument("--query_csv_path", required=True, type=str,
                    help="csv containing query image data")
parser.add_argument("--gallery_csv_path", required=True, type=str,
                    help="csv containing gallery image data")
# dataset root dizini (CSV içindeki path'ler bu köke göre relatif)
parser.add_argument("--data_dir", type=str, required=True,
                    help="root directory for image datasets")
# modelin beklediği input görüntü boyutu
parser.add_argument("--input_size", type=int, default=224,
                    help="Image input size for the model")
parser.add_argument("--batchsize", type=int, default=64)
# kaç tane gallery görüntüsü gösterilecek (query + num_images)
parser.add_argument("--num_images", type=int, default=40,
                    help="number of gallery images to show")
parser.add_argument("--imgs_per_row", type=int, default=6)
# Daha önce test.py tarafından üretilmiş .mat dosyasını kullanıp kullanmama seçeneği
parser.add_argument("--use_saved_mat", action="store_true",
                    help="Query ve Gallery feature'ları için daha önce oluşturulmuş .mat dosyası, tekrar hesaplama yapmamıza gerek kalmaz.")
args = parser.parse_args()

# CUDA varsa kullan, yoksa CPU kullan
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
h, w = args.input_size, args.input_size  # yükseklik ve genişlik

######################################################################
# Load Data (veri ve dönüşümler)
######################################################################

# Görüntü ön işleme: resize, tensor'a çevir, normalize
data_transforms = transforms.Compose([
    transforms.Resize((h, w), interpolation=3),  # interpolation=3 ~ PIL.Image.BICUBIC
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet normalizasyonu
])

# CSV'leri oku: query ve gallery (her satırda path ve id)
query_df = pd.read_csv(args.query_csv_path)
gallery_df = pd.read_csv(args.gallery_csv_path)

# classes: query ve gallery'deki benzersiz ID'lerin birleşimi(tüm araçların id'lerini elde ettik)
classes = list(pd.concat([query_df["id"], gallery_df["id"]]).unique())

# use_cam: eğer hem query hem gallery CSV'lerinde "cam" sütunu varsa true
use_cam = "cam" in query_df.columns and "cam" in gallery_df.columns

# ImageDataset sınıfı: CSV'yi kullanarak PyTorch Dataset oluşturur (senin projendeki implementasyona bağlı)
image_datasets = {
    "query": ImageDataset(args.data_dir, query_df, "id", classes, transform=data_transforms),
    "gallery": ImageDataset(args.data_dir, gallery_df, "id", classes, transform=data_transforms),
}


# DataLoader'lar (batch halinde feature çıkarmak için). shuffle=False olmalı, çünkü test-sıralaması sabit olmalı.
dataloaders = {
    x: torch.utils.data.DataLoader(
        image_datasets[x], batch_size=args.batchsize, shuffle=False, num_workers=2
    ) for x in ["gallery", "query"]
}

######################################################################
# Feature Extraction (modelden embedding alma fonksiyonları)
######################################################################

def fliplr(img):
    """
    Aynı aracın simetrik görünümünden de özellik çıkar.
    Çünkü bir aracı hem soldan hem sağdan görebilirsin.
    Bu fonksiyon input'u (N, C, H, W) (batch, kanal, yükseklik, genişlik) şeklinde bekler.
    """
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long().to(img.device)
    return img.index_select(3, inv_idx)


def extract_features(model, dataloader):
    """
    Bir dataloader'daki tüm görüntüler için modelden feature çıkarır.
    - model: remove_classifier=True argümanı ile yüklenmiş, sınıflandırma katmanı kaldırılmış model
    - dataloader: DataLoader (gallery veya query için)
    Döndürür: features (N x D) torch tensor ve labels (numpy array)
    """
    img_count = 0
    # dummy batch ile feature boyutunu öğren (model(dummy).shape => (batch, feature_dim))
    dummy = next(iter(dataloader))[0].to(device)
    output = model(dummy)
    feature_dim = output.shape[1]
    labels = []

    for idx, data in enumerate(tqdm.tqdm(dataloader)):
        X, y = data  # X: batch görüntü tensorleri, y: batch etiketleri (id)
        n, c, h, w = X.size()
        img_count += n
        # ff: bu batch için feature'ların toplanacağı tensor
        ff = torch.FloatTensor(n, feature_dim).zero_().to(device)

        # batch etiketlerini listeye ekle
        for lab in y:
            labels.append(lab)

        # flip augmentasyonu kullanarak hem orijinal hem flip feature'ı topla
        for i in range(2):
            if i == 1:
                X = fliplr(X)
            input_X = Variable(X.to(device))
            outputs = model(input_X)  # (batch, feature_dim)
            ff += outputs

        # L2 normalize (satır satır)
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)  # (batch,1)
        ff = ff.div(fnorm.expand_as(ff))

        # features tensor'unu ilk batch için oluştur
        if idx == 0:
            features = torch.FloatTensor(len(dataloader.dataset), ff.shape[1])

        # batch'in features'ını global features tensora yerleştir
        start = idx * args.batchsize
        end = min((idx + 1) * args.batchsize, len(dataloader.dataset))
        features[start:end, :] = ff
        print("DEBUG - feature vs label alignment kontrolü")
        for i in range(5):
            print(f"Index {i}: label={labels[i]}, path={dataloader.dataset.df.iloc[i]['path']}")

    return features, np.array(labels)  # features: (N, D) numpy değil, torch tensor; labels: numpy array


def extract_feature(model, img):
    """
    Tek bir görüntü için feature çıkarır (query için kullanılıyor).
    - img: transform uygulanmış tek görüntü tensoru (C,H,W) veya (1,C,H,W)
    Döndürür: normalize edilmiş feature (torch tensor, 1D)
    """
    if len(img.shape) == 3:
        img = torch.unsqueeze(img, 0)  # -> (1,C,H,W)
    img = img.to(device)
    feature = model(img).reshape(-1)  # model çıktısını 1D vektöre çevir

    # flip et, ekle, normalize et — böylece model flip'e duyarsızlaşır
    img = fliplr(img)
    flipped_feature = model(img).reshape(-1)
    feature += flipped_feature

    fnorm = torch.norm(feature, p=2)
    return feature.div(fnorm)  # normalize edilmiş 1D tensor döner


def get_scores(query_feature, gallery_features):
    """
    Query feature ile tüm gallery feature'lar arasındaki benzerlik skorlarını hesaplar.
    - query_feature: 1D (D,) veya torch tensor
    - gallery_features: (N, D) torch tensor (L2-normalize edilmiş olmalı)
    Burada dot product kullanılıyor; eğer feature'lar L2-normalize ise dot == cosine similarity.
    Döndürür: numpy array (N,) skorlar (büyük olan daha benzer)
    """
    query = query_feature.view(-1, 1)  # (D,1)
    score = torch.mm(gallery_features, query)  # (N,1) = her gallery ile dot product
    score = score.squeeze(1).cpu().numpy()  # (N,)
    return score


def show_query_result(axes, query_img, gallery_imgs, query_label, gallery_labels):
    query_trans = transforms.Pad(4, 0)
    good_trans = transforms.Pad(4, (0, 255, 0))
    bad_trans = transforms.Pad(4, (255, 0, 0))

    for idx, img in enumerate([query_img] + gallery_imgs):
        img = img.resize((128, 128))
        if idx == 0:
            img = query_trans(img)
            label = int(query_label)  # doğrudan gerçek ID
        else:
            correct = int(query_label) == int(gallery_labels[idx - 1])
            img = good_trans(img) if correct else bad_trans(img)
            label = int(gallery_labels[idx - 1])  # doğrudan gerçek ID

        ax = axes.flat[idx]
        ax.imshow(img)
        ax.set_title(f"ID: {label}", fontsize=8, pad=2)

    for ax in axes.flat:
        ax.axis("off")


######################################################################
# Run Queries (ana akış)
######################################################################

# Eğer --use_saved_mat verilmişse, test.py tarafından oluşturulmuş 'pytorch_result.mat' dosyasını yükle
if args.use_saved_mat:
    saved_res = scipy.io.loadmat("pytorch_result.mat")
    # .mat içinde 'gallery_f', 'gallery_label', 'query_f', 'query_label' bekleniyor
    gallery_features = torch.Tensor(saved_res["gallery_f"])  # (N_gallery, D)
    gallery_labels = saved_res["gallery_label"].reshape(-1)  # (N_gallery,)
    query_features = torch.Tensor(saved_res["query_f"])     # (N_query, D)
    query_labels = saved_res["query_label"].reshape(-1)     # (N_query,)
else:
    # model'i opts ve checkpoint ile yükle (sınıf katmanı kaldırılmış şekilde)
    model = load_model_from_opts(args.model_opts, args.checkpoint, remove_classifier=True)
    model.eval()
    model.to(device)

    print("Computing gallery features ...")
    # gallery için tüm feature'ları çıkar (uzun sürebilir)
    with torch.no_grad():
        gallery_features, gallery_labels = extract_features(model, dataloaders["gallery"])
        # gallery_features: torch tensor (N_gallery, D)
        # gallery_labels: numpy array (N_gallery,)

# query dataset objesi
dataset = image_datasets["query"]
# queries: dataset indexlerinin listesi (0..N-1)
queries = list(range(len(dataset)))
random.shuffle(queries)  # sıra karıştırılır; ama kullanmak için aşağıda queries kullanılıyor
# curr_idx: hangi pozisyondayız (karışık listedeki indeks)
curr_idx = random.randint(0, len(queries) - 1)  # rastgele başlat

def on_key(event):
    """
    Klavye olayları (sol/sağ/enter)
    - left: önceki query
    - right: sonraki query
    - enter: mevcut figürü PDF'e kaydet
    """
    global curr_idx
    if event.key == "left":
        curr_idx = (curr_idx - 1) % len(queries)
    elif event.key == "right":
        curr_idx = (curr_idx + 1) % len(queries)
    elif event.key == "enter":
        fig.savefig("reid_query_result.pdf", pad_inches=0, bbox_inches='tight')
    else:
        return
    refresh_plot()  # yeni curr_idx'e göre plot'u yenile

def refresh_plot():
    """
    Mevcut query(index) için skorları hesaplar ve figure'ı günceller.
    Önemli: q_index = queries[curr_idx] — random shuffle edilmiş sıra kullanılmaktadır.
    """
    global curr_idx
    q_index = queries[curr_idx]  # gerçek dataset index'i

    # Eğer saved_mat kullanılıyorsa query feature'lar önceden var, yoksa anlık hesapla
    if args.use_saved_mat:
        q_feature = query_features[q_index]  # (D,)
        y = query_labels[q_index]            # sorgunun id'si
    else:
        X, y = dataset[q_index]  # dataset'ten (transformed_tensor, id) al
        with torch.no_grad():
            q_feature = extract_feature(model, X).cpu()  # anlık feature çıkar ve CPU'ya al

    # Eğer 'cam' bilgisi varsa aynı kameradan gelen gallery örneklerini dışla
    if use_cam:
        curr_cam = query_df["cam"].iloc[q_index]
        # good_gallery_idx: True olanlar (farklı kameralar) — aynı cam'ler False olacak
        good_gallery_idx = torch.tensor(gallery_df["cam"] != curr_cam).type(torch.bool)
        # gallery_orig_idx: orijinal gallery indeksleri (boolean mask -> positions)
        gallery_orig_idx = np.where(good_gallery_idx)[0]
        # gal_features: sadece farklı kameradan alınmış feature'lar
        gal_features = gallery_features[good_gallery_idx]
    else:
        # eğer cam yoksa tüm gallery'i kullan
        gallery_orig_idx = np.arange(len(gallery_df))
        gal_features = gallery_features

    # skorları hesapla (benzerlik puanları)
    gallery_scores = get_scores(q_feature, gal_features)
    # azalan sıraya göre index al (en yüksek skor en başta)
    idx = np.argsort(gallery_scores)[::-1]

    # g_labels: sıralanmış gallery id'leri (kullanıcıya gösterilen sıra için)
    if use_cam:
        g_labels = gallery_labels[gallery_orig_idx][idx]
    else:
        g_labels = gallery_labels[idx]

    # görselleri al: query resmi ve en yakın num_images gallery resimleri
    q_img = dataset.get_image(q_index)
    g_imgs = [image_datasets["gallery"].get_image(gallery_orig_idx[i])
              for i in idx[:args.num_images]]

    # görselleştir
    show_query_result(axes, q_img, g_imgs, y, g_labels)
    fig.canvas.draw()
    fig.canvas.flush_events()

# subplot'ları oluştur (satır sayısı = (1 + num_images) / imgs_per_row aşağı yuvarlanır)
n_rows = math.ceil((1 + args.num_images) / args.imgs_per_row)
fig, axes = plt.subplots(n_rows, args.imgs_per_row, figsize=(12, 15))
fig.canvas.mpl_connect("key_press_event", on_key)  # klavye event'lerini bağla

# Başlangıç görüntüsünü çiz
refresh_plot()
plt.show()


print("Classes:", classes)
print("Query IDs:", query_df["id"].unique())
print("Gallery IDs:", gallery_df["id"].unique())
