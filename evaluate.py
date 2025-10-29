import scipy.io
import torch
import numpy as np
import argparse

parser = argparse.ArgumentParser(
    description="Evaluate precomputed query and gallery features.")
parser.add_argument("--gpu", action="store_true", help="Use gpu")
parser.add_argument("--no_cams", action="store_true",
                    help="dont remove samples with same id and same cam as the query from the gallery.")
parser.add_argument("--K", type=int, default=-1,
                    help="If provided mAP@K will be calculated, else the same range will be used.")
args = parser.parse_args()

#######################################################################
# Evaluate

def evaluate(qf, ql, gf, gl, qc=None, gc=None, K=100):
    """
    Tek bir query için:
      - gf @ qf benzerlik (cosine varsayımı: embedding'ler L2-normalize ise)
      - Skorları azalan sırada diz
      - good_index: aynı ID (ql) olan galeriler
      - junk_index: (1) etiket < 0 olanlar + (2) aynı ID ve aynı CAM olanlar (kamera filtresi açıksa)
      - İsteğe bağlı K kesmesi (mAP@K / CMC@K)
    """
    # Skor: (N_galeri, D) x (D, 1) -> (N_galeri, 1)
    query = qf.view(-1, 1)
    score = torch.mm(gf, query).squeeze(1).cpu().numpy()

    # Skoru azalan sırala (en benzerler başa)
    index = np.argsort(score)[::-1]

    # Aynı ID olanlar (good) ve etiket < 0 olanlar (junk, yoksa boş)
    good_index = np.where(gl == ql)[0]
    junk_index = np.where(gl < 0)[0]

    # Kamera filtresi: aynı ID + aynı CAM -> junk
    if qc is not None and gc is not None:
        camera_index = np.where(gc == qc)[0]                       # aynı kamera olan galeri pozisyonları
        camera_junk = np.intersect1d(camera_index, good_index, assume_unique=True)
        junk_index = np.concatenate([junk_index, camera_junk])     # junk'a ekle
        # good: aynı id olup junk olmayanlar
        good_index = np.setdiff1d(good_index, camera_index, assume_unique=True)

    # K sınırı (mAP@K / CMC@K)
    if 0 < K < len(index):
        index = index[:K]
        good_index = np.intersect1d(index, good_index, assume_unique=True)
        junk_index = np.intersect1d(index, junk_index, assume_unique=True)

    return compute_mAP(index, good_index, junk_index)


def compute_mAP(index, good_index, junk_index):
    """
    mAP ve CMC hesaplar.
    - index: skorla sıralanmış galeri indeksleri (büyükten küçüğe)
    - good_index: doğru eşleşmelerin galeri indeksleri
    - junk_index: değerlendirme dışı (çıkarılacak) galeri indeksleri
    Dönüş:
      ap (float), cmc (IntTensor[K])
    """
    ap = 0.0
    cmc = torch.zeros(len(index), dtype=torch.int32)

    # Hiç doğru yoksa (pozitif yok), protokole göre bu sorgu mAP/CMC’ye sayılmaz.
    if good_index.size == 0:
        cmc[0] = -1
        return ap, cmc

    # Junk’ları listeden çıkar
    mask_valid = ~np.in1d(index, junk_index)
    index = index[mask_valid]

    # Good’ların bu sıralamadaki konumları
    mask_good = np.in1d(index, good_index)
    rows_good = np.flatnonzero(mask_good)   # 0, 5, 12, ...

    # CMC: ilk doğru bulunduğu yerden itibaren 1
    cmc[rows_good[0]:] = 1

    # AP: doğru bulunan her pozisyonda precision ortalaması (11-pt değil, integral approx.)
    ngood = len(good_index)
    for i, r in enumerate(rows_good, start=1):
        # i: şimdiye kadar bulunan doğru sayısı, r: sıralamadaki pozisyon (0-index)
        precision = i / (r + 1)
        old_precision = (i - 1) / r if r != 0 else 1.0
        ap += (1.0 / ngood) * (old_precision + precision) / 2.0

    return float(ap), cmc


######################################################################
device = torch.device("cuda") if args.gpu else torch.device("cpu")

result = scipy.io.loadmat('pytorch_result.mat')
query_feature = torch.FloatTensor(result['query_f'])
query_label = result['query_label'][0]
gallery_feature = torch.FloatTensor(result['gallery_f'])
gallery_label = result['gallery_label'][0]

query_cam = result['query_cam'].reshape(-1)
gallery_cam = result['gallery_cam'].reshape(-1)
use_cam = (len(gallery_cam) > 0 and len(query_cam) > 0) and not args.no_cams

query_feature = query_feature.to(device)
gallery_feature = gallery_feature.to(device)

K = args.K if args.K >= 1 else len(gallery_label)
CMC = torch.IntTensor(K).zero_()
ap = 0.0

for i in range(len(query_label)):
    qc = query_cam[i] if use_cam else None
    gc = gallery_cam if use_cam else None
    ap_tmp, CMC_tmp = evaluate(
        query_feature[i], query_label[i], gallery_feature, gallery_label,
        qc, gc, K
    )
    if CMC_tmp[0] == -1:
        continue
    CMC = CMC + CMC_tmp
    ap += ap_tmp

CMC = CMC.float()
CMC = CMC / len(query_label)  # average CMC
print('Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f' %
      (CMC[0], CMC[4], CMC[9], ap / len(query_label)))