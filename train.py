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

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import yaml
from shutil import copyfile
import pandas as pd
import tqdm

from pytorch_metric_learning import losses, miners

version = list(map(int, torch.__version__.split(".")[:2]))
torchvision_version = list(map(int, torchvision.__version__.split(".")[:2]))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

from random_erasing import RandomErasing
from circle_loss import CircleLoss, convert_label_to_similarity
from instance_loss import InstanceLoss
from load_model import load_model_from_opts
from dataset import ImageDataset, BatchSampler

######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--data_dir', required=True,
                    type=str, help='path to the dataset root directory')
parser.add_argument("--train_csv_path", required=True, type=str)
parser.add_argument("--val_csv_path", required=True, type=str)
parser.add_argument('--name', default='ft_ResNet50',
                    type=str, help='output model name')

parser.add_argument('--gpu_ids', default='0', type=str,
                    help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--num_workers', default=2, type=int)

parser.add_argument('--warm_epoch', default=0, type=int,
                    help='the first K epoch that needs warm up (counted from start_epoch)')
parser.add_argument('--total_epoch', default=60,
                    type=int, help='total training epoch')
parser.add_argument("--save_freq", default=5, type=int,
                    help="frequency of saving the model in epochs")
parser.add_argument("--checkpoint", default="", type=str,
                    help="Model checkpoint to load.")
parser.add_argument("--start_epoch", default=0, type=int,
                    help="Epoch to continue training from.")

parser.add_argument('--fp16', action='store_true',
                    help='Use mixed precision training.')

parser.add_argument("--grad_clip_max_norm", type=float, default=50.0,
                    help="maximum norm of gradient to be clipped to")

parser.add_argument('--lr', default=0.05,
                    type=float, help='base learning rate for the head. 0.1 * lr is used for the backbone')
parser.add_argument('--cosine', action='store_true',
                    help='use cosine learning rate')
parser.add_argument('--batchsize', default=32,
                    type=int, help='batchsize')
parser.add_argument('--linear_num', default=512, type=int,
                    help='feature dimension: 512 (default) or 0 (linear=False)')
parser.add_argument('--stride', default=2, type=int, help='last stride')
parser.add_argument('--droprate', default=0.5,
                    type=float, help='drop rate')
parser.add_argument('--erasing_p', default=0.5, type=float,
                    help='Random Erasing probability, in [0,1]')
parser.add_argument('--color_jitter', action='store_true',
                    help='use color jitter in training')
parser.add_argument("--label_smoothing", default=0.0, type=float)
parser.add_argument("--samples_per_class", default=1, type=int,
                    help="Batch sampling strategy. Batches are sampled from groups of the same class with *this many* elements, if possible. Ordinary random sampling is achieved by setting this to 1.")

parser.add_argument("--model", default="resnet_ibn",
                    help="""what model to use, supported values: ['resnet', 'resnet_ibn', 'densenet', 'swin',
                    'NAS', 'hr', 'efficientnet'] (default: resnet_ibn)""")
parser.add_argument("--model_subtype", default="default",
                    help="Subtype for the model (b0 to b7 for efficientnet, '50'/'101'/'152' for resnet)")
parser.add_argument("--mixstyle", action="store_true",
                    help="Use MixStyle in training for domain generalization (only for resnet variants yet)")

# Metric-learning losses
parser.add_argument('--arcface', action='store_true',
                    help='use ArcFace loss')
parser.add_argument('--circle', action='store_true',
                    help='use Circle loss')
parser.add_argument('--cosface', action='store_true',
                    help='use CosFace loss')
parser.add_argument('--contrast', action='store_true',
                    help='use supervised contrastive loss')
parser.add_argument('--instance', action='store_true',
                    help='use instance loss')
parser.add_argument('--ins_gamma', default=32, type=int,
                    help='gamma for instance loss')
parser.add_argument('--triplet', action='store_true',
                    help='use triplet loss')
parser.add_argument('--lifted', action='store_true',
                    help='use lifted loss')
parser.add_argument('--sphere', action='store_true',
                    help='use sphere loss')

# Debug
parser.add_argument("--debug", action="store_true")
parser.add_argument("--debug_period", type=int, default=100,
                    help="Print the loss and grad statistics for *this many* batches at a time.")

# Early stopping
parser.add_argument("--early_stop_patience", type=int, default=0,
                    help="Number of epochs with no val improvement after which training will be stopped. 0 disables early stopping.")
parser.add_argument("--early_stop_min_delta", type=float, default=0.0,
                    help="Minimum change in val metric to qualify as an improvement for early stopping.")

opt = parser.parse_args()

# Fix label smoothing version check: only warn for torch 1.x < 1.10
if opt.label_smoothing > 0.0 and (version[0] == 1 and version[1] < 10):
    warnings.warn(
        "Label smoothing is supported only from torch 1.10.0, the parameter will be ignored"
    )

######################################################################
# Configure devices
# ---------
fp16 = opt.fp16
data_dir = opt.data_dir
name = opt.name

gpu_ids = []
if opt.gpu_ids:
    str_ids = opt.gpu_ids.split(',')
    for str_id in str_ids:
        gid = int(str_id)
        if gid >= 0:
            gpu_ids.append(gid)

use_gpu = torch.cuda.is_available() and len(gpu_ids) > 0
if not use_gpu:
    print("Running on CPU ...")
    device = torch.device("cpu")
else:
    print("Running on cuda:{}".format(gpu_ids[0]))
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True
    device = torch.device("cuda")

######################################################################
# Load Data
# ---------
h, w = 224, 224
interpolation = (
    3 if torchvision_version[0] == 0 and torchvision_version[1] < 13
    else transforms.InterpolationMode.BICUBIC
)

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

if opt.erasing_p > 0:
    transform_train_list.append(
        RandomErasing(probability=opt.erasing_p, mean=[0.0, 0.0, 0.0])
    )

if opt.color_jitter:
    transform_train_list = [transforms.ColorJitter(
        brightness=0.1, contrast=0.1, saturation=0.1, hue=0)] + transform_train_list

print("Train transforms:", transform_train_list)
data_transforms = {
    'train': transforms.Compose(transform_train_list),
    'val': transforms.Compose(transform_val_list),
}

train_df = pd.read_csv(opt.train_csv_path)
val_df = pd.read_csv(opt.val_csv_path)
all_ids = list(set(train_df["id"]).union(set(val_df["id"])))
image_datasets = {
    "train": ImageDataset(opt.data_dir, train_df, "id", classes=all_ids, transform=data_transforms["train"]),
    "val": ImageDataset(opt.data_dir, val_df, "id", classes=all_ids, transform=data_transforms["val"]),
}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
opt.nclasses = len(class_names)
print("Number of classes in total: {}".format(opt.nclasses))

######################################################################
# Utilities
# ---------
class DebugInfo:
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


# histories
y_loss = {'train': [], 'val': []}
y_err = {'train': [], 'val': []}

def fliplr(img):
    """flip a batch of images horizontally"""
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long().to(img.device)
    img_flip = img.index_select(3, inv_idx)
    return img_flip

######################################################################
# Training function
# ------------------
def train_model(model, criterion, start_epoch=0, num_epochs=25, num_workers=2):
    since = time.time()

    model = model.to(device)

    scaler = torch.amp.GradScaler('cuda', enabled=fp16)

    # optimizer & scheduler
    optim_name = optim.SGD
    ignored_params = list(map(id, model.classifier.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    classifier_params = model.classifier.parameters()

    optimizer = optim_name(
        [
            {'params': base_params, 'initial_lr': 0.1 * opt.lr, 'lr': 0.1 * opt.lr},
            {'params': classifier_params, 'initial_lr': opt.lr, 'lr': opt.lr},
        ],
        weight_decay=5e-4,
        momentum=0.9,
        nesterov=True
    )

    if opt.cosine:
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=opt.total_epoch, eta_min=0.01 * opt.lr
        )
    else:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # fast-forward scheduler if continuing from checkpoint
    for _ in range(start_epoch):
        scheduler.step()

    # warmup
    warm_up = 0.1  # start from 0.1 * lr
    if opt.warm_epoch > 0:
        warm_iteration = max(
            1, round(dataset_sizes['train'] / opt.batchsize) * opt.warm_epoch
        )
    else:
        warm_iteration = 0

    # metric-learning losses
    if opt.arcface:
        criterion_arcface = losses.ArcFaceLoss(
            num_classes=opt.nclasses, embedding_size=512).to(device)
    else:
        criterion_arcface = None

    if opt.cosface:
        criterion_cosface = losses.CosFaceLoss(
            num_classes=opt.nclasses, embedding_size=512).to(device)
    else:
        criterion_cosface = None

    if opt.circle:
        criterion_circle = CircleLoss(m=0.25, gamma=32).to(device)
    else:
        criterion_circle = None

    if opt.triplet:
        miner = miners.MultiSimilarityMiner()
        criterion_triplet = losses.TripletMarginLoss(margin=0.3).to(device)
    else:
        miner = None
        criterion_triplet = None

    if opt.lifted:
        criterion_lifted = losses.GeneralizedLiftedStructureLoss(
            neg_margin=1, pos_margin=0).to(device)
    else:
        criterion_lifted = None

    if opt.contrast:
        criterion_contrast = losses.ContrastiveLoss(
            pos_margin=0, neg_margin=1).to(device)
    else:
        criterion_contrast = None

    if opt.instance:
        criterion_instance = InstanceLoss(gamma=opt.ins_gamma).to(device)
    else:
        criterion_instance = None

    if opt.sphere:
        criterion_sphere = losses.SphereFaceLoss(
            num_classes=opt.nclasses, embedding_size=512, margin=4).to(device)
    else:
        criterion_sphere = None

    # dataloaders
    train_sampler = BatchSampler(
        image_datasets["train"], opt.batchsize, opt.samples_per_class
    )

    dataloaders = {
        "train": torch.utils.data.DataLoader(
            image_datasets["train"],
            batch_sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=use_gpu
        ),
        "val": torch.utils.data.DataLoader(
            image_datasets["val"],
            batch_size=opt.batchsize,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=use_gpu
        )
    }

    # early stopping setup (based on best val accuracy)
    best_val_acc = 0.0
    best_epoch = start_epoch
    epochs_no_improve = 0
    stop_early = False

    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            if opt.debug:
                loss_debug = DebugInfo("loss", opt.debug_period)
                grad_debug = DebugInfo("grad", opt.debug_period)
            else:
                loss_debug = None
                grad_debug = None

            running_loss = torch.zeros(1, device=device)
            running_corrects = torch.zeros(1, device=device)

            for data in tqdm.tqdm(dataloaders[phase]):
                inputs, labels = data
                now_batch_size = inputs.size(0)

                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.amp.autocast('cuda', enabled=fp16):
                    outputs = model(inputs)

                    # if we requested features (metric losses)
                    if return_feature:
                        logits, ff = outputs
                        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
                        ff = ff.div(fnorm.expand_as(ff))

                        loss = criterion(logits, labels)
                        _, preds = torch.max(logits.data, 1)

                        if criterion_arcface is not None:
                            loss = loss + criterion_arcface(ff, labels) / now_batch_size
                        if criterion_cosface is not None:
                            loss = loss + criterion_cosface(ff, labels) / now_batch_size
                        if criterion_circle is not None:
                            loss = loss + criterion_circle(
                                *convert_label_to_similarity(ff, labels)
                            ) / now_batch_size
                        if criterion_triplet is not None and miner is not None:
                            hard_pairs = miner(ff, labels)
                            loss = loss + criterion_triplet(ff, labels, hard_pairs)
                        if criterion_lifted is not None:
                            loss = loss + criterion_lifted(ff, labels)
                        if criterion_contrast is not None:
                            loss = loss + criterion_contrast(ff, labels)
                        if criterion_instance is not None:
                            loss = loss + criterion_instance(ff, labels) / now_batch_size
                        if criterion_sphere is not None:
                            loss = loss + criterion_sphere(ff, labels) / now_batch_size
                    else:
                        logits = outputs
                        _, preds = torch.max(logits.data, 1)
                        loss = criterion(logits, labels)

                    if loss_debug is not None:
                        loss_debug.step(loss.item())

                    # warmup (only in train phase)
                    if phase == 'train' and warm_iteration > 0 and epoch < opt.warm_epoch:
                        warm_up = min(1.0, warm_up + 0.9 / warm_iteration)
                        loss = loss * warm_up

                # backward + optimize
                if phase == 'train':
                    scaler.scale(loss).backward()

                    # grad clipping
                    scaler.unscale_(optimizer)
                    old_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), opt.grad_clip_max_norm
                    )
                    if grad_debug is not None:
                        grad_debug.step(old_norm.item())

                    scaler.step(optimizer)
                    scaler.update()

                running_loss += loss.item() * now_batch_size
                running_corrects += float(torch.sum(preds == labels.data))

            epoch_loss = running_loss.cpu() / dataset_sizes[phase]
            epoch_acc = running_corrects.cpu() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss.item(), epoch_acc.item()))

            y_loss[phase].append(epoch_loss)
            y_err[phase].append(1.0 - epoch_acc)

            # validation: save checkpoints & early stopping
            if phase == 'val':
                # periodic checkpoint
                if epoch == num_epochs - 1 or (epoch % opt.save_freq) == (opt.save_freq - 1):
                    save_network(model, epoch)

                # curve
                draw_curve(epoch)

                # early stopping logic
                if opt.early_stop_patience > 0:
                    if epoch_acc.item() > best_val_acc + opt.early_stop_min_delta:
                        best_val_acc = epoch_acc.item()
                        best_epoch = epoch
                        epochs_no_improve = 0
                        # save best model
                        save_network(model, 'best')
                    else:
                        epochs_no_improve += 1
                        print(
                            f"No improvement in val acc for {epochs_no_improve} epoch(s). "
                            f"Best so far: {best_val_acc:.4f} at epoch {best_epoch}"
                        )
                        if epochs_no_improve >= opt.early_stop_patience:
                            print("Early stopping triggered.")
                            stop_early = True

            # train phase: step LR scheduler
            if phase == 'train':
                scheduler.step()

        time_elapsed = time.time() - since
        print('Epoch complete at {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print()

        if stop_early:
            break

    total_time = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        total_time // 60, total_time % 60))

    return model

######################################################################
# Curve drawing
# ---------------------------
fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")

def draw_curve(current_epoch):
    ax0.clear()
    ax1.clear()
    ax0.set_title("loss")
    ax1.set_title("top1err")
    ax0.plot([e for e in range(len(y_loss["train"]))],
             [l.item() for l in y_loss['train']], 'bo-', label='train')
    ax0.plot([e for e in range(len(y_loss["val"]))],
             [l.item() for l in y_loss['val']], 'ro-', label='val')
    ax1.plot([e for e in range(len(y_err["train"]))],
             [e_.item() for e_ in y_err['train']], 'bo-', label='train')
    ax1.plot([e for e in range(len(y_err["val"]))],
             [e_.item() for e_ in y_err['val']], 'ro-', label='val')
    if current_epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig(os.path.join(SCRIPT_DIR, "model", name, 'train.jpg'))

######################################################################
# Save model
# ---------------------------
def save_network(network, epoch_label):
    save_filename = f'net_{epoch_label}.pth'
    save_path = os.path.join(SCRIPT_DIR, "model", name, save_filename)
    current_device = next(iter(network.parameters())).device
    torch.save(network.cpu().state_dict(), save_path)
    network.to(current_device)

######################################################################
# Save opts and load model
# ---------------------------
model_root = os.path.join(SCRIPT_DIR, "model")
os.makedirs(model_root, exist_ok=True)
dir_name = os.path.join(model_root, name)
os.makedirs(dir_name, exist_ok=True)

# record every run
copyfile(os.path.join(SCRIPT_DIR, 'train.py'),
         os.path.join(dir_name, "train.py"))
copyfile(os.path.join(SCRIPT_DIR, "model.py"),
         os.path.join(dir_name, "model.py"))

# save opts
opts_file = "%s/opts.yaml" % dir_name
with open(opts_file, 'w') as fp:
    yaml.dump(vars(opt), fp, default_flow_style=False)

# if any metric loss is active, we need features
return_feature = (
    opt.arcface or opt.cosface or opt.circle or opt.triplet or
    opt.contrast or opt.instance or opt.lifted or opt.sphere
)

model = load_model_from_opts(
    opts_file,
    ckpt=opt.checkpoint if opt.checkpoint else None,
    return_feature=return_feature
)
model.train()

######################################################################
# Train and evaluate
# ---------------------------
if version[0] > 1 or (version[0] == 1 and version[1] >= 10):
    criterion = torch.nn.CrossEntropyLoss(
        label_smoothing=opt.label_smoothing
    )
else:
    criterion = torch.nn.CrossEntropyLoss()

model = train_model(
    model, criterion,
    start_epoch=opt.start_epoch,
    num_epochs=opt.total_epoch,
    num_workers=opt.num_workers
)
