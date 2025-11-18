import os
import sys
import yaml
import torch
from torch import nn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

from model import (
    ft_net,
    ft_net_dense,
    ft_net_hr,
    ft_net_swin,
    ft_net_efficient,
    ft_net_NAS,
    PCB,
)

sys.path.remove(SCRIPT_DIR)


def load_weights(model, ckpt_path):
    """
    Checkpoint'ten weight yükler.
    state_dict içindeki isimler modelle tam uyuşmasa bile strict=False ile yükleriz.
    """
    state = torch.load(ckpt_path, map_location="cpu")

    # Eski hack'leri unutalım; classifier şekli tutmuyorsa zaten remove_classifier ile atacağız.
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state, strict=False)
    return model


def create_model(n_classes, kind="resnet", **kwargs):
    if kind == "resnet":
        return ft_net(n_classes, **kwargs)
    elif kind == "densenet":
        return ft_net_dense(n_classes, **kwargs)
    elif kind == "hr":
        return ft_net_hr(n_classes, **kwargs)
    elif kind == "efficientnet":
        return ft_net_efficient(n_classes, **kwargs)
    elif kind == "NAS":
        return ft_net_NAS(n_classes, **kwargs)
    elif kind == "swin":
        return ft_net_swin(n_classes, **kwargs)
    elif kind == "PCB":
        return PCB(n_classes)
    else:
        raise ValueError(f"Model type cannot be created: {kind}")


def load_model(
    n_classes,
    kind="resnet",
    ckpt=None,
    remove_classifier=False,
    **kwargs,
):
    model = create_model(n_classes, kind, **kwargs)
    if ckpt:
        load_weights(model, ckpt)

    if remove_classifier and hasattr(model, "classifier"):
        # Ft-net yapılarında classifier genelde ClassBlock içinde
        if hasattr(model.classifier, "classifier"):
            model.classifier.classifier = nn.Sequential()
        else:
            # Yine de sıradan bir Sequential ise komple boşalt
            model.classifier = nn.Sequential()
        model.eval()

    return model


def load_model_from_opts(
    opts_file,
    ckpt=None,
    return_feature=False,
    remove_classifier=False,
):
    with open(opts_file, "r") as f:
        opts = yaml.load(f, Loader=yaml.FullLoader)

    n_classes = opts["nclasses"]
    droprate = opts["droprate"]
    stride = opts["stride"]
    linear_num = opts["linear_num"]

    model_subtype = opts.get("model_subtype", "default")
    model_type = opts.get("model", "resnet_ibn")
    mixstyle = opts.get("mixstyle", False)

    # ------------------------------
    # MODEL SEÇİMİ
    # ------------------------------
    if model_type in ("resnet", "resnet_ibn"):
        model = create_model(
            n_classes,
            "resnet",
            droprate=droprate,
            ibn=(model_type == "resnet_ibn"),
            stride=stride,
            circle=return_feature,
            linear_num=linear_num,
            model_subtype=model_subtype,
            mixstyle=mixstyle,
        )

    elif model_type == "densenet":
        model = create_model(
            n_classes,
            "densenet",
            droprate=droprate,
            circle=return_feature,
            linear_num=linear_num,
        )

    elif model_type == "efficientnet":
        model = create_model(
            n_classes,
            "efficientnet",
            droprate=droprate,
            circle=return_feature,
            linear_num=linear_num,
            model_subtype=model_subtype,
        )

    elif model_type == "NAS":
        model = create_model(
            n_classes,
            "NAS",
            droprate=droprate,
            linear_num=linear_num,
        )

    elif model_type == "PCB":
        model = create_model(n_classes, "PCB")

    elif model_type == "hr":
        model = create_model(
            n_classes,
            "hr",
            droprate=droprate,
            circle=return_feature,
            linear_num=linear_num,
        )

    elif model_type == "swin":
        # 🔥 En kritik yer: Eğitimde ft_net_swin kullanıldı, burada da onu kullanacağız.
        model = create_model(
            n_classes,
            "swin",
            droprate=droprate,
            stride=stride,
            circle=return_feature,
            linear_num=linear_num,
            model_subtype=model_subtype,
        )

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # ------------------------------
    # WEIGHT YÜKLE
    # ------------------------------
    if ckpt:
        load_weights(model, ckpt)

    # ------------------------------
    # EMBEDDING MODU
    # ------------------------------
    if remove_classifier and hasattr(model, "classifier"):
        if hasattr(model.classifier, "classifier"):
            model.classifier.classifier = nn.Sequential()
        else:
            model.classifier = nn.Sequential()
        model.eval()

    return model
