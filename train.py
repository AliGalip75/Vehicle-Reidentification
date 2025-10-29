# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import time
import os
import sys
import warnings

import torch
import torch.optim as optim
import torch.cuda.amp as amp
from torch.optim import lr_scheduler
import torchvision
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

# Görselleştirme backend'ini dosyaya yazmak için ayarla (GUI gerektirmez)
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import yaml
from shutil import copyfile
import pandas as pd
import numpy as np
import tqdm

# Metrik öğrenme kayıpları ve miner'lar
from pytorch_metric_learning import losses, miners

# Torch ve torchvision sürüm bilgileri
version = list(map(int, torch.__version__.split(".")[:2]))
torchvision_version = list(map(int, torchvision.__version__.split(".")[:2]))

# Proje kökünü import yoluna ekle
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

# Yerel yardımcılar / kayıplar / veri kümesi
from random_erasing import RandomErasing
from circle_loss import CircleLoss, convert_label_to_similarity
from instance_loss import InstanceLoss
from load_model import load_model_from_opts
from dataset import ImageDataset, BatchSampler


######################################################################
# Argümanlar / Seçenekler
# ------------------------
parser = argparse.ArgumentParser(
    description='Training'
)

#? Veri yolu ve CSV dosyaları
parser.add_argument(
    '--data_dir', 
    required=True,
    type=str, help='veri kümesi kök dizini'
)
parser.add_argument(
    "--train_csv_path", 
    required=True, type=str,
    help='eğitim CSV yolu'
)
parser.add_argument(
    "--val_csv_path", 
    required=True, 
    type=str,
    help='doğrulama CSV yolu'
)

#? Model isimlendirme
parser.add_argument(
    '--name', 
    default='ft_ResNet50',
    type=str, 
    help='çıktı model klasör adı'
)

#? Donanım ayarları
parser.add_argument(
    '--gpu_ids', 
    default='0', 
    type=str,
    help='GPU id listesi: "0" veya "0,1,2" gibi'
)
parser.add_argument(
    '--tpu_cores', 
    default=-1, 
    type=int,
    help="GPU yerine TPU kullan (çekirdek sayısı)."
)
parser.add_argument(
    '--num_workers', 
    default=2, 
    type=int,
    help="DataLoader işçi sayısı"
)

#? Eğitim döngüsü ayarları
parser.add_argument(
    '--warm_epoch', 
    default=0, 
    type=int,
    help='başlangıçta ısınma yapılacak epoch sayısı (start_epoch’tan itibaren)'
)
parser.add_argument(
    '--total_epoch', 
    default=60,
    type=int, 
    help='toplam eğitim epoch sayısı'
)
parser.add_argument(
    "--save_freq", 
    default=5, 
    type=int,
    help="model kaydetme sıklığı (epoch)"
)
parser.add_argument(
    "--checkpoint", 
    default="", 
    type=str,
    help="yüklenmek istenen model checkpoint yolu"
)
parser.add_argument(
    "--start_epoch", 
    default=0, 
    type=int,
    help="eğitime devam ederken başlangıç epoch’u"
)

#? Karma (mixed) hassasiyet ve gradyan kırpma
parser.add_argument(
    '--fp16', 
    action='store_true',
    help='mixed precision (fp16) eğitim kullan'
)
parser.add_argument(
    "--grad_clip_max_norm", 
    type=float, 
    default=50.0,
    help="gradyan kırpma için maksimum norm"
)

#? Optimizasyon ve mimari ile ilgili
parser.add_argument(
    '--lr', 
    default=0.05,
    type=float, 
    help='head için temel öğrenme oranı (backbone için bunun 0.1 katı kullanılır)'
)
parser.add_argument(
    '--cosine', 
    action='store_true',
    help='cosine LR scheduleri kullan'
)
parser.add_argument(
    '--batchsize', 
    default=32,
    type=int, 
    help='batch boyutu'
)
parser.add_argument(
    '--linear_num', 
    default=512, 
    type=int,
    help='özellik boyutu: 512 (varsayılan) veya 0 (linear=False)'
)
parser.add_argument(
    '--stride', 
    default=2, 
    type=int,
    help='son katmanda stride'
)
parser.add_argument(
    '--droprate', 
    default=0.5,
    type=float, 
    help='dropout oranı'
)
parser.add_argument(
    '--erasing_p', 
    default=0.5, 
    type=float,
    help='Random Erasing olasılığı, [0,1]'
)
parser.add_argument(
    '--color_jitter', 
    action='store_true',
    help='eğitimde color jitter kullan'
)
parser.add_argument(
    "--label_smoothing", 
    default=0.0, 
    type=float,
    help="label smoothing katsayısı"
)

# ? Batch örnekleme stratejisi
parser.add_argument(
    "--samples_per_class",
    default=1,
    type=int,
    help="Batch örnekleme: mümkünse her sınıftan bu kadar örnek alarak batch oluştur. 1 => klasik random sampling."
)

# ? Model seçimi
parser.add_argument(
    "--model",
    default="resnet_ibn",
    help="kullanılacak model: ['resnet','resnet_ibn','densenet','swin','NAS','hr','efficientnet']"
)
parser.add_argument(
    "--model_subtype",
    default="default",
    help="alt tip (ör. efficientnet için b0..b7)"
)
parser.add_argument(
    "--mixstyle",
    action="store_true",
    help="Domain genelleme için MixStyle kullan (şimdilik resnet varyantları)"
)

# ? Kayıp fonksiyon bayrakları (ek/alternatif kayıplar)
parser.add_argument(
    "--arcface",
    action="store_true",
    help="ArcFace kaybı kullan"
)
parser.add_argument(
    "--circle",
    action="store_true",
    help="CircleLoss kullan"
)
parser.add_argument(
    "--cosface",
    action="store_true",
    help="CosFace kaybı kullan"
)
parser.add_argument(
    "--contrast",
    action="store_true",
    help="Supervised Contrastive Loss kullan"
)
parser.add_argument(
    "--instance",
    action="store_true",
    help="Instance loss kullan"
)
parser.add_argument(
    "--ins_gamma",
    default=32,
    type=int,
    help="instance loss gamma"
)
parser.add_argument(
    "--triplet",
    action="store_true",
    help="Triplet loss kullan"
)
parser.add_argument(
    "--lifted",
    action="store_true",
    help="Lifted loss kullan"
)
parser.add_argument(
    "--sphere",
    action="store_true",
    help="SphereFace loss kullan"
)

# ? Debug çıktıları
parser.add_argument(
    "--debug",
    action="store_true",
    help="debug çıktıları aktif"
)
parser.add_argument(
    "--debug_period",
    type=int,
    default=100,
    help="her bu kadar batch’te bir istatistik yazdır"
)
opt = parser.parse_args()

#-------------------------------

# Label smoothing desteği yoksa uyar
if opt.label_smoothing > 0.0 and version[0] < 1 or version[1] < 10:
    warnings.warn("!!!!!!!!!!!!!!!!!!!!!!!!!!Label smoothing yalnızca torch 1.10.0+ ile desteklenir; parametre yok sayılacak!!!!!!!!!!!!!!!!!!!!!!!!!!!!")


################################
# Cihaz / Donanım yapılandırması
# ------------------------------
fp16 = opt.fp16
data_dir = opt.data_dir
name = opt.name

if opt.tpu_cores > 0:
    # TPU ile çalışılacaksa
    use_tpu, use_gpu = True, False
    print("Running on TPU ...")
else:
    # GPU id’lerini çöz
    gpu_ids = []
    if opt.gpu_ids:
        str_ids = opt.gpu_ids.split(',')
        for str_id in str_ids:
            gid = int(str_id)
            if gid >= 0:
                gpu_ids.append(gid)

    # TPU iptal edildi, GPU kontrol edildi, GPU yoksa CPU ayarlandı
    use_tpu = False
    use_gpu = torch.cuda.is_available() and len(gpu_ids) > 0
    if not use_gpu:
        print("Running on CPU ...")
    else:
        print("Running on cuda:{}".format(gpu_ids[0]))
        torch.cuda.set_device(gpu_ids[0])
        cudnn.benchmark = True

####################
# Veri ve Dönüşümler
# ------------------
# Eğitim ve doğrulama için temel dönüşümler
h, w = 224, 224
interpolation = 3 if torchvision_version[0] == 0 and torchvision_version[1] < 13 else \
    transforms.InterpolationMode.BICUBIC

# Eğitim ve doğrulama dönüşüm listeleri (ayrı)
transform_train_list = [
    transforms.Resize((h, w), interpolation=interpolation),
    transforms.Pad(10),
    transforms.RandomCrop((h, w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]
transform_val_list = [
    transforms.Resize(size=(h, w), interpolation=interpolation),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

# Random Erasing (opsiyonel)
if opt.erasing_p > 0:
    transform_train_list = transform_train_list + [RandomErasing(probability=opt.erasing_p, mean=[0.0, 0.0, 0.0])]

# Color Jitter (opsiyonel)
if opt.color_jitter:
    transform_train_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)] + transform_train_list

print("Train transforms List:", transform_train_list)

# Eğitim ve doğrulama dönüşüm listeleri (sözlük)
data_transforms = {
    'train': transforms.Compose(transform_train_list),
    'val': transforms.Compose(transform_val_list),
}

# CSV'den veri setlerini oluştur
image_datasets = {}

train_df = pd.read_csv(opt.train_csv_path)
val_df = pd.read_csv(opt.val_csv_path)

all_ids = list(set(train_df["id"]).union(set(val_df["id"])))

image_datasets["train"] = ImageDataset(
    opt.data_dir, train_df, "id", classes=all_ids, transform=data_transforms["train"])
image_datasets["val"] = ImageDataset(
    opt.data_dir, val_df, "id", classes=all_ids, transform=data_transforms["val"])

# Dataset büyüklükleri ve sınıf sayısı
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
opt.nclasses = len(class_names)
print("Toplam Sınıf Sayısı: {}".format(opt.nclasses))

#############################
# Eğitim yardımcıları (debug)
# ---------------------------
class DebugInfo:
    """Periyodik olarak metrik istatistikleri yazdırmak için yardımcı."""
    def __init__(self, name, print_period):
        self.history = []
        self.name = name
        self.print_period = print_period

    def step(self, value):
        self.history.append(value)
        if len(self.history) >= self.print_period:
            print("\n{}:".format(self.name))
            print(pd.Series(self.history).describe())
            self.history = []


#####################################
# Kayıp ve hata takibi için geçmişler
# -----------------------------------
y_loss = {'train': [], 'val': []}  # loss geçmişi
y_err = {'train': [], 'val': []}    # hata geçmişi (kullanılmıyor ama çizimde yeri ayrılmış)

# Kullanmadık sanırım
def fliplr(img):
    """Bir görüntü batch'ini yatay eksende çevirir (N,C,H,W)."""
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long().to(img.device)
    img_flip = img.index_select(3, inv_idx)
    return img_flip


# --------------------------------------------------
# Basit Re-ID metrikleri (Rank@K ve mAP) hesaplayıcı
# --------------------------------------------------
def compute_reid_metrics(features, labels, topk=(1, 5, 10)):
    """
    Amaç:
      - Her örneği bir "sorgu" kabul edip, geri kalan örnekler
        arasından en benzerleri bulmak.
      - "Doğru eşleşme": aynı ID'ye sahip örnekler.
      - Rank@K: Sorgu için ilk K sonuç içinde en az 1 doğru eşleşme var mı?
      - mAP: Doğru eşleşmeler ne kadar yukarıda? (ortalama hassasiyet)

    Parametreler:
      features : (N, D) torch.Tensor -> modelden çıkan embedding'ler
      labels   : (N,)   torch.Tensor -> her embedding'in kimlik etiketi
      topk     : ölçmek istediğin K'ler (örn. 1,5,10)

    Dönüş:
      (rank1, rank5, rank10, mAP)  -> topk neyse ona göre döner
    """

    # 1) Özellikleri normalize et (birim uzunluk). Böylece mesafe/benzerlik daha tutarlı olur.
    features = F.normalize(features, p=2, dim=1)

    # 2) Tüm örnekler arası L2 mesafe matrisi: (N x N)
    # Küçük mesafe daha benzer demektir.
    dist = torch.cdist(features, features, p=2)

    # 3) Kendi kendine eşleşmeyi kapat:
    # Diyagonali çok büyük bir değere çekerek "sorgu kendini bulmasın" diyoruz.
    dist.fill_diagonal_(1e9)

    N = labels.size(0)
    # Rank@K sayaçları (kaç sorguda ilk K içinde en az 1 doğru bulduk?)
    correct = {k: 0 for k in topk}
    # Ortalama AP için her sorgunun AP'sini toplayacağız
    APs = []

    # 4) Her sorgu (i) için...
    for i in range(N):
        # Bu sorgunun galeriye olan mesafeleri
        distances = dist[i]  # (N,)

        # Hangi örnekler aynı kimlikte? (pozitifler True)
        is_pos = (labels == labels[i])  # (N,) bool

        # 5) En yakınlardan uzağa doğru sırala (küçük mesafe önce)
        sorted_idx = torch.argsort(distances)         # (N,)
        sorted_is_pos = is_pos[sorted_idx].cpu().numpy()  # numpy'a geçtik (kolay hesap için)

        # 6) CMC / Rank@K: ilk K içinde en az 1 True var mı?
        for k in topk:
            if np.any(sorted_is_pos[:k]):
                correct[k] += 1

        # 7) AP: True olanların sıralamadaki indekslerinden hesaplanır.
        #    True'ların indekslerini al -> her birinde "i / rank" şeklinde precision değerlerini al -> ortalamasını al.
        if np.any(sorted_is_pos):
            hit_positions = np.where(sorted_is_pos == 1)[0]        # doğru eşleşmelerin pozisyonları (0-index)
            precision_at_hits = np.arange(1, len(hit_positions)+1) / (hit_positions + 1)
            APs.append(np.mean(precision_at_hits))
        # Not: Hiç pozitif yoksa o sorgu mAP'e dahil edilmez (yaygın uygulama)

    # 8) Rank@K'ler: doğru bulunan sorgu sayısını toplam sorguya böl
    # (pozitifi olmayan sorgular da rank hesaplamasında paydada kalır)
    rank1  = correct.get(1,  0) / N
    rank5  = correct.get(5,  0) / N
    rank10 = correct.get(10, 0) / N

    # 9) mAP: AP'lerin ortalaması (pozitifi olan sorgular üzerinden)
    mAP = float(np.mean(APs)) if APs else 0.0

    return rank1, rank5, rank10, mAP

# ---------------
# Eğitim döngüsü
# ---------------
def train_model(model, criterion, start_epoch=0, num_epochs=25, num_workers=2):
    since = time.time()

    device = torch.device("cuda" if use_gpu else "cpu")
    model = model.to(device)

    # Mixed precision için scaler ve autocast
    if fp16:
        scaler = amp.GradScaler("cuda" if use_gpu else "cpu")
        autocast = amp.autocast("cuda" if use_gpu else "cpu")

    # Optimizatör ve LR scheduler
    optim_name = optim.SGD
    ignored_params = list(map(id, model.classifier.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    classifier_params = model.classifier.parameters()

    optimizer = optim_name([
        {'params': base_params, 'initial_lr': 0.1 * opt.lr, 'lr': 0.1 * opt.lr},  # backbone
        {'params': classifier_params, 'initial_lr': opt.lr, 'lr': opt.lr},        # classifier/head
    ], weight_decay=5e-4, momentum=0.9, nesterov=True)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    if opt.cosine:
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, opt.total_epoch, eta_min=0.01 * opt.lr)

    # start_epoch > 0 ise scheduleri ileri sar
    for _ in range(start_epoch):
        scheduler.step()

    # Dataloader'lar (train için sınıf-dengeli BatchSampler)
    train_sampler = BatchSampler(image_datasets["train"], opt.batchsize, opt.samples_per_class)
    dataloaders = {
        "val": torch.utils.data.DataLoader(image_datasets["val"], batch_size=opt.batchsize,
                                           num_workers=num_workers, pin_memory=use_gpu),
        "train": torch.utils.data.DataLoader(image_datasets["train"], batch_sampler=train_sampler,
                                             num_workers=num_workers, pin_memory=use_gpu)
    }

    for epoch in range(start_epoch, num_epochs):
        print(f"Epoch {epoch}/{num_epochs-1}")
        print('-' * 10)

        for phase in ['train', 'val']:
            loader = tqdm.tqdm(dataloaders[phase])
            model.train(phase == 'train')  # train/val modu

            running_loss = torch.zeros(1).to(device)

            # Doğrulamada metrik hesaplamak için tüm özellikleri topla
            all_features = []
            all_labels = []

            for data in loader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                # İleri geçiş (val’de gradsız)
                if phase == 'val':
                    with torch.no_grad():
                        outputs = model(inputs)
                else:
                    if fp16:
                        autocast.__enter__()
                    outputs = model(inputs)

                # Çıktı düzeni: return_feature True ise (logits, feat) beklenir
                if return_feature:
                    logits, ff = outputs
                    fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
                    ff = ff.div(fnorm.expand_as(ff))
                    loss = criterion(logits, labels)
                else:
                    loss = criterion(outputs, labels)
                    ff = outputs  # feat olarak logits kullanılıyor (sınıflandırma bazlı eğitim)

                running_loss += loss.item() * inputs.size(0)

                # Geri yayılım ve optimizasyon (sadece train’de)
                if phase == 'train':
                    if fp16:
                        autocast.__exit__(None, None, None)
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()

                # Doğrulamada özellik ve etiket topla (Re-ID metrikleri için)
                if phase == 'val':
                    all_features.append(ff.detach().cpu())
                    all_labels.append(labels.detach().cpu())

            # Epoch kaybını yazdır
            epoch_loss = running_loss.cpu().item() / dataset_sizes[phase]
            print(f'{phase} Loss: {epoch_loss:.4f}')
            y_loss[phase].append(epoch_loss)

            # 🔹 Doğrulama metrikleri: Rank@K ve mAP
            if phase == 'val':
                if len(all_features) > 0:
                    feats = torch.cat(all_features, dim=0)
                    lbls = torch.cat(all_labels, dim=0)
                    rank1, rank5, rank10, mAP = compute_reid_metrics(feats, lbls)
                    print(f'Validation ReID -> Rank@1: {rank1:.4f}, Rank@5: {rank5:.4f}, Rank@10: {rank10:.4f}, mAP: {mAP:.4f}')

                # Modeli periyodik kaydet ve eğriyi çiz
                if epoch == num_epochs - 1 or (epoch % opt.save_freq == (opt.save_freq - 1)):
                    save_network(model, epoch)
                    draw_curve(epoch)

            if phase == 'train':
                scheduler.step()

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')
    return model


def tpu_map_fn(index, flags):
    """ TPU süreçleri için iş parçacığı başlangıç fonksiyonu (örnek şablon). """
    torch.manual_seed(flags["seed"])
    if version[0] > 1 or (version[0] == 1 and version[1] >= 10):
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=opt.label_smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    train_model(model, criterion, opt.start_epoch, opt.total_epoch, num_workers=flags["num_workers"])


######################################################################
# Eğri çizimi (loss/err)
# ----------------------
x_epoch = []
fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")


def draw_curve(current_epoch):
    """Loss ve (varsa) hata eğrilerini kaydeder."""
    x_epoch.append(current_epoch)
    ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
    ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
    ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
    ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')
    if current_epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig(os.path.join(SCRIPT_DIR, "model", name, 'train.jpg'))

######################################################################
# Model kaydetme
# --------------
def save_network(network, epoch_label):
    """Ağı (state_dict) belirtilen epoch etiketiyle diske kaydeder."""
    save_filename = 'net_%s.pth' % epoch_label
    save_path = os.path.join(SCRIPT_DIR, "model", name, save_filename)
    device = next(iter(network.parameters())).device
    torch.save(network.cpu().state_dict(), save_path)
    network.to(device)


######################################################################
# Çalışma klasörü ve seçeneklerin kaydı
# -------------------------------------
dir_name = os.path.join(SCRIPT_DIR, "model", name)
if not os.path.isdir(dir_name):
    os.mkdir(dir_name)

# Her çalışmayı kayıt altına al (script ve model tanımı)
copyfile(os.path.join(SCRIPT_DIR, 'train.py'), os.path.join(dir_name, "train.py"))
copyfile(os.path.join(SCRIPT_DIR, "model.py"), os.path.join(dir_name, "model.py"))

# Opts'leri yaml olarak kaydet
opts_file = "%s/opts.yaml" % dir_name
with open(opts_file, 'w') as fp:
    yaml.dump(vars(opt), fp, default_flow_style=False)

# Ek kayıplar kullanılıyorsa modelin özellik (feat) döndürmesini iste
return_feature = (
    opt.arcface or opt.cosface or opt.circle or opt.triplet or
    opt.contrast or opt.instance or opt.lifted or opt.sphere
)

# Modeli seçeneklerden yükle (checkpoint varsa kullan)
model = load_model_from_opts(opts_file,
                             ckpt=opt.checkpoint if opt.checkpoint else None,
                             return_feature=return_feature)
# Cihaz ataması train_model içinde yapılacak
model.train()


######################################################################
# Eğitim ve doğrulama başlatma
# ----------------------------
if use_tpu and opt.tpu_cores > 1:
    # Çok çekirdekli TPU örneği (şablon; içerikler yorumlanmış)
    pass
else:
    # Kayıp fonksiyonu (label smoothing destekliyse etkin)
    if version[0] > 1 or (version[0] == 1 and version[1] >= 10):
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=opt.label_smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    # Eğitimi başlat
    model = train_model(
        model, criterion, start_epoch=opt.start_epoch, num_epochs=opt.total_epoch,
        num_workers=opt.num_workers
    )
