"""
Train the baseline miner (MobileNetV2 classifier-only) on the MVTec bottle
category and export the result as an ONNX model.

Usage
-----
    python training/train_baseline.py \
        --data   data/datasets/MVTec_AD \
        --out    training/models/baseline.onnx \
        --epochs 20 \
        --lr     1e-4 \
        --batch  16 \
        --device cpu        # or "cuda"

The script:
  1. Loads good images from bottle/train/good  (label=0)
     and defective images from bottle/test/{broken_large,broken_small,contamination} (label=1)
  2. Applies the same augmentation pipeline used by the validator
     (flip, rotation, crop, photometric jitter) so the model generalises
     to the validator's test images.
  3. Fine-tunes BaselineMiner with binary cross-entropy.
  4. Exports to ONNX with fixed input shape [1, 3, 256, 256].
"""

import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Allow running from repo root or from training/
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from architectures.baseline import BaselineMiner

# ─────────────────────────────────────────────────────────────────── dataset ──

DEFECT_CATS = ["broken_large", "broken_small", "contamination"]
IMG_EXTS    = {".jpg", ".jpeg", ".png", ".bmp"}


def _collect_samples(mvtec_root: str):
    """Return list of (path, label) pairs: label 0 = good, 1 = defective."""
    bottle = os.path.join(mvtec_root, "bottle")
    samples = []

    good_dir = os.path.join(bottle, "train", "good")
    for f in sorted(os.listdir(good_dir)):
        if os.path.splitext(f)[1].lower() in IMG_EXTS:
            samples.append((os.path.join(good_dir, f), 0))

    for cat in DEFECT_CATS:
        cat_dir = os.path.join(bottle, "test", cat)
        for f in sorted(os.listdir(cat_dir)):
            if os.path.splitext(f)[1].lower() in IMG_EXTS:
                samples.append((os.path.join(cat_dir, f), 1))

    return samples


class BottleDataset(Dataset):
    """MVTec bottle classification dataset with augmentation."""

    # ImageNet normalisation
    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    def __init__(self, samples, augment: bool = True):
        self.samples = samples
        if augment:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(15),
                transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
                transforms.ToTensor(),
                transforms.Normalize(self.MEAN, self.STD),
                transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 1.5))], p=0.5),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize(self.MEAN, self.STD),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), torch.tensor(label, dtype=torch.float32)


# ───────────────────────────────────────────────────────────────── training ──

def train(args):
    device = torch.device(args.device)

    # ── Data ──────────────────────────────────────────────────────────────────
    samples = _collect_samples(args.data)
    random.seed(42)
    random.shuffle(samples)

    n_val    = max(1, int(0.15 * len(samples)))
    val_samp = samples[:n_val]
    trn_samp = samples[n_val:]

    # Balance classes in training set by oversampling the minority
    pos = [s for s in trn_samp if s[1] == 1]
    neg = [s for s in trn_samp if s[1] == 0]
    if len(pos) < len(neg):
        pos = pos * (len(neg) // len(pos) + 1)
        pos = pos[:len(neg)]
    trn_samp = pos + neg
    random.shuffle(trn_samp)

    train_loader = DataLoader(
        BottleDataset(trn_samp, augment=True),
        batch_size=args.batch, shuffle=True, num_workers=0, pin_memory=False,
    )
    val_loader = DataLoader(
        BottleDataset(val_samp, augment=False),
        batch_size=args.batch, shuffle=False, num_workers=0,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model     = BaselineMiner(freeze_backbone=False).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"Training baseline on {len(trn_samp)} images, validating on {len(val_samp)}")
    print(f"Device: {device}  |  Epochs: {args.epochs}  |  Batch: {args.batch}")

    best_val_loss = float("inf")
    best_state    = None

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        trn_loss, correct, total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            conf = model(imgs)
            loss = criterion(conf, labels)
            loss.backward()
            optimizer.step()
            trn_loss += loss.item() * len(imgs)
            correct  += ((conf >= 0.5).float() == labels).sum().item()
            total    += len(imgs)
        scheduler.step()

        # Validate
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                conf      = model(imgs)
                val_loss += criterion(conf, labels).item() * len(imgs)
                val_correct += ((conf >= 0.5).float() == labels).sum().item()
                val_total   += len(imgs)

        trn_acc = correct  / total
        val_acc = val_correct / val_total
        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"trn_loss={trn_loss/total:.4f}  trn_acc={trn_acc:.3f}  "
            f"val_loss={val_loss/val_total:.4f}  val_acc={val_acc:.3f}"
        )

        if val_loss / val_total < best_val_loss:
            best_val_loss = val_loss / val_total
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # ── Export ────────────────────────────────────────────────────────────────
    model.load_state_dict(best_state)
    model.eval().cpu()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    dummy = torch.zeros(1, 3, 256, 256)
    torch.onnx.export(
        model, dummy, args.out,
        input_names=["image"],
        output_names=["confidence"],
        dynamic_axes={"image": {0: "batch"}, "confidence": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    print(f"\nSaved ONNX model → {args.out}")


# ──────────────────────────────────────────────────────────────── entrypoint ──

def _parse():
    p = argparse.ArgumentParser(description="Train baseline miner")
    p.add_argument("--data",   default="data/datasets/MVTec_AD", help="MVTec_AD root")
    p.add_argument("--out",    default="training/models/baseline.onnx", help="Output ONNX path")
    p.add_argument("--epochs", type=int,   default=20)
    p.add_argument("--lr",     type=float, default=1e-4)
    p.add_argument("--batch",  type=int,   default=16)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


if __name__ == "__main__":
    train(_parse())
