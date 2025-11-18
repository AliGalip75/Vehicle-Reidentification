# reid/metrics.py

import numpy as np
import torch

from typing import Optional

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

    # AP: doğru bulunan her pozisyonda precision ortalaması
    ngood = len(good_index)
    for i, r in enumerate(rows_good, start=1):
        precision = i / (r + 1)
        old_precision = (i - 1) / r if r != 0 else 1.0
        ap += (1.0 / ngood) * (old_precision + precision) / 2.0

    return float(ap), cmc


def evaluate_single(qf, ql, gf, gl, qc=None, gc=None, K=100):
    """
    Tek bir query için:
      - gf @ qf benzerlik (cosine varsayımı: embedding'ler L2-normalize ise)
      - Skorları azalan sırada diz
      - good_index: aynı ID (ql) olan galeriler
      - junk_index: (1) etiket < 0 olanlar + (2) aynı ID ve aynı CAM olanlar (kamera filtresi açıksa)
      - İsteğe bağlı K kesmesi (mAP@K / CMC@K)
    """
    query = qf.view(-1, 1)
    score = torch.mm(gf, query).squeeze(1).cpu().numpy()

    # Skoru azalan sırala (en benzerler başa)
    index = np.argsort(score)[::-1]

    # Aynı ID olanlar (good) ve etiket < 0 olanlar (junk, yoksa boş)
    good_index = np.where(gl == ql)[0]
    junk_index = np.where(gl < 0)[0]

    # Kamera filtresi: aynı ID + aynı CAM -> junk
    if qc is not None and gc is not None:
        camera_index = np.where(gc == qc)[0]
        camera_junk = np.intersect1d(camera_index, good_index, assume_unique=True)
        junk_index = np.concatenate([junk_index, camera_junk])
        good_index = np.setdiff1d(good_index, camera_index, assume_unique=True)

    # K sınırı (mAP@K / CMC@K)
    if 0 < K < len(index):
        index = index[:K]
        good_index = np.intersect1d(index, good_index, assume_unique=True)
        junk_index = np.intersect1d(index, junk_index, assume_unique=True)

    return compute_mAP(index, good_index, junk_index)


def compute_reid_metrics(
    query_features: torch.Tensor,
    query_labels,
    gallery_features: torch.Tensor,
    gallery_labels,
    query_cams=None,
    gallery_cams=None,
    use_cam: bool = True,
    K: Optional[int] = None
):
    """
    evaluate.py'deki ana döngünün DRF versiyonu.
    query_features: (Nq, D)  (CPU veya GPU)
    gallery_features: (Ng, D)
    query_labels, gallery_labels: 1D numpy array veya list
    query_cams, gallery_cams: isteğe bağlı kamera id dizileri
    """
    device = query_features.device
    query_features = query_features.to(device)
    gallery_features = gallery_features.to(device)

    query_labels = np.array(query_labels)
    gallery_labels = np.array(gallery_labels)

    if query_cams is not None:
        query_cams = np.array(query_cams)
    if gallery_cams is not None:
        gallery_cams = np.array(gallery_cams)

    if K is None:
        K = gallery_features.size(0)

    CMC = torch.IntTensor(K).zero_()
    ap_sum = 0.0
    num_queries = len(query_labels)

    for i in range(num_queries):
        qc = query_cams[i] if (use_cam and query_cams is not None and gallery_cams is not None) else None
        gc = gallery_cams if (use_cam and query_cams is not None and gallery_cams is not None) else None

        ap_tmp, CMC_tmp = evaluate_single(
            query_features[i],
            query_labels[i],
            gallery_features,
            gallery_labels,
            qc,
            gc,
            K,
        )
        if CMC_tmp[0] == -1:
            # Bu query için hiç "good" yok, evaluate.py'deki gibi atla
            continue

        CMC = CMC + CMC_tmp
        ap_sum += ap_tmp

    CMC = CMC.float()
    CMC = CMC / num_queries

    metrics = {
        "K": int(K),
        "rank1": float(CMC[0]),
        "rank5": float(CMC[4]) if K >= 5 else None,
        "rank10": float(CMC[9]) if K >= 10 else None,
        "mAP": float(ap_sum / num_queries),
    }
    return metrics
