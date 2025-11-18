# reid/engine.py

import os
from threading import Lock

import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from PIL import Image
import pandas as pd
from torch.utils.data import DataLoader

from django.conf import settings

from load_model import load_model_from_opts
from dataset import ImageDataset

from typing import Optional

torchvision_version = list(map(int, torchvision.__version__.split(".")[:2]))


class ReidEngine:
    def __init__(
        self,
        opts_path: str,
        checkpoint_path: str,
        data_root: str,
        gallery_csv_path: str,
        batch_size: int = 64,
        num_workers: int = 0,
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = load_model_from_opts(
            opts_file=opts_path,
            ckpt=checkpoint_path,
            remove_classifier=True,
        )
        self.model = self.model.to(self.device)    # 🔥 EKLENDİ
        self.model.eval()

        h, w = 224, 224
        interpolation = (
            3
            if torchvision_version[0] == 0 and torchvision_version[1] < 13
            else transforms.InterpolationMode.BICUBIC
        )

        self.transform = transforms.Compose(
            [
                transforms.Resize((h, w), interpolation=interpolation),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225],
                ),
            ]
        )

        self.data_root = data_root
        self.gallery_csv_path = gallery_csv_path
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.gallery_df = None
        self.gallery_features = None
        self.gallery_ids = None
        self.gallery_cams = None

        self._build_gallery()

    # ----------------------------------------------------------------------
    def _build_gallery(self):
        df = pd.read_csv(self.gallery_csv_path)
        self.gallery_df = df

        self.gallery_ids = df["id"].to_numpy()
        self.gallery_cams = df["cam"].to_numpy() if "cam" in df.columns else None
        self.gallery_paths = df["path"].tolist()

        dataset = ImageDataset(
            img_root=self.data_root,
            df=df,
            target_label="id",
            classes="infer",
            transform=self.transform,
        )

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        feats = []
        self.model.eval()

        with torch.no_grad():
            for images, _ in loader:
                print("Batch:", images.shape)
                images = images.to(self.device)   # 🔥 GPU/CPU mismatch fix

                emb = self.model(images)          # (B, D) olmalı
                print("EMB SHAPE:", emb.shape)

                emb = F.normalize(emb, p=2, dim=1)
                feats.append(emb.cpu())

        if len(feats) == 0:
            raise RuntimeError("Gallery features boş, dataset / CSV yanlış olabilir.")

        self.gallery_features = torch.cat(feats, dim=0)  # (Ng, D)
        print("GALLERY FEATURES SHAPE:", self.gallery_features.shape)


    # ----------------------------------------------------------------------
    def extract_feature_from_pil(self, image: Image.Image) -> torch.Tensor:
        self.model.eval()
        img = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.model(img)                # (1, D)
            emb = F.normalize(emb, p=2, dim=1)
        return emb.cpu()


    # ----------------------------------------------------------------------
    def search(self, image: Image.Image, top_k: int = 10):

        if self.gallery_features is None:
            raise RuntimeError("Gallery features henüz yüklenmemiş.")

        qf = self.extract_feature_from_pil(image).squeeze(0)

        gf = self.gallery_features
        scores = torch.mv(gf, qf)

        top_k = min(top_k, gf.size(0))
        values, indices = torch.topk(scores, k=top_k, largest=True)

        results = []
        for rank, (score, idx) in enumerate(zip(values.tolist(), indices.tolist()), start=1):
            row = self.gallery_df.iloc[idx]
            results.append(
                {
                    "rank": rank,
                    "score": float(score),
                    "id": int(row["id"]),
                    "cam": int(row["cam"]) if "cam" in row and not pd.isna(row["cam"]) else None,
                    "path": row["path"],
                }
            )
        return results


_engines = {}
_engine_lock = Lock()


def get_reid_engine(model_name: str):
    global _engines
    with _engine_lock:
        if model_name not in _engines:
            cfg = settings.REID_MODELS[model_name]
            _engines[model_name] = ReidEngine(
                opts_path=str(cfg["opts"]),
                checkpoint_path=str(cfg["ckpt"]),
                data_root=str(settings.REID_DATA_ROOT),
                gallery_csv_path=str(settings.REID_GALLERY_CSV),
                batch_size=settings.REID_BATCH_SIZE,
                num_workers=settings.REID_NUM_WORKERS,
            )
        return _engines[model_name]
