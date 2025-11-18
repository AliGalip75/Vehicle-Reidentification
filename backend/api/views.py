# reid/views.py

from io import BytesIO

from PIL import Image
import pandas as pd
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, permissions

from django.conf import settings

from dataset import ImageDataset
from .reid_engine import get_reid_engine
from .metrics import compute_reid_metrics


class SearchView(APIView):
    """
    POST /api/search/
    Form-data:
      - image: query görseli
      - top_k (opsiyonel, default: 10)
    """
    permission_classes = [permissions.AllowAny]

    def post(self, request, *args, **kwargs):
        if "image" not in request.FILES:
            return Response(
                {"detail": "image alanı zorunlu."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        top_k = int(request.data.get("top_k", 10))

        file_obj = request.FILES["image"]
        try:
            image = Image.open(file_obj).convert("RGB")
        except Exception as e:
            return Response(
                {"detail": f"Görüntü açılamadı: {e}"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        model_name = request.data.get("model", "resnet")
        engine = get_reid_engine(model_name)
        results = engine.search(image, top_k=top_k)

        return Response({
            "model": model_name,
            "top_k": top_k,
            "results": results,
        })



class MetricsView(APIView):
    permission_classes = [permissions.AllowAny]

    def get(self, request, *args, **kwargs):
        model_name = request.query_params.get("model", "resnet")
        cfg = settings.REID_MODELS[model_name]

        metrics_path = cfg["metrics"]

        import json
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        return Response(metrics, status=status.HTTP_200_OK)

