"""
Train the improved miner (MobileNetV2 + U-Net decoder, dual heads) on the
MVTec bottle category and export the result as an ONNX model.

Usage
-----
    python training/train_improved.py \
        --data   data/datasets/MVTec_AD \
        --out    training/models/improved.onnx \
        --epochs 20 \
        --lr     1e-4 \
        --batch  8 \
        --device cpu        # or "cuda"

The improved model trains with a combined loss:

    L = α_cls * BCE(conf, label) + α_seg * (BCE(mask_pred, mask_gt) + Dice(mask_pred, mask_gt))

Ground-truth masks come from bottle/ground_truth/{broken_large,...}/<stem>_mask.png.
Good images have an all-zero mask target.

The same augmentation (flip, rotation, crop, photometric jitter) is applied
identically to both the image and its mask to preserve pixel correspondence.
"""

import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from architectures.improved import ImprovedMiner

# ─────────────────────────────────────────────────────────────────── dataset ──

DEFECT_CATS = ["broken_large", "broken_small", "contamination"]
IMG_EXTS    = {".jpg", ".jpeg", ".png", ".bmp"}


def _collect_samples(mvtec_root: str):
    """
    Return list of (img_path, mask_path_or_None, label) tuples.
    Good images have mask_path=None and an all-zero target mask.
    """
    bottle  = os.path.join(mvtec_root, "bottle")
    samples = []

    good_dir = os.path.join(bottle, "train", "good")
    for f in sorted(os.listdir(good_dir)):
        if os.path.splitext(f)[1].lower() in IMG_EXTS:
            samples.append((os.path.join(good_dir, f), None, 0))

    for cat in DEFECT_CATS:
        cat_dir = os.path.join(bottle, "test", cat)
        gt_dir  = os.path.join(bottle, "ground_truth", cat)
        for f in sorted(os.listdir(cat_dir)):
            if os.path.splitext(f)[1].lower() in IMG_EXTS:
                stem      = os.path.splitext(f)[0]
                mask_path = os.path.join(gt_dir, f"{stem}_mask.png")
                samples.append((os.path.join(cat_dir, f), mask_path, 1))

    return samples


class _PairedAugment:
    """Apply identical geometric augmentation to an image+mask pair."""

    def __init__(self, augment: bool):
        self.augment = augment

    def __call__(self, img: Image.Image, mask: Image.Image):
        if not self.augment:
            img  = img.resize((256, 256), Image.BILINEAR)
            mask = mask.resize((256, 256), Image.NEAREST)
            return img, mask

        # Random horizontal flip
        if random.random() < 0.5:
            img  = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        # Random vertical flip
        if random.random() < 0.5:
            img  = img.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        # Random rotation ±15°
        angle = random.uniform(-15, 15)
        img   = img.rotate(angle,  resample=Image.BILINEAR, expand=False)
        mask  = mask.rotate(angle, resample=Image.NEAREST,  expand=False)

        # Random crop 80–100% then resize to 256×256
        w, h       = img.size
        scale      = random.uniform(0.8, 1.0)
        crop_w, crop_h = int(w * scale), int(h * scale)
        x0 = random.randint(0, w - crop_w)
        y0 = random.randint(0, h - crop_h)
        box  = (x0, y0, x0 + crop_w, y0 + crop_h)
        img  = img.crop(box).resize((256, 256), Image.BILINEAR)
        mask = mask.crop(box).resize((256, 256), Image.NEAREST)

        return img, mask


MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

_to_tensor_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
_photometric = transforms.Compose([
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 1.5))], p=0.5),
])


class BottleSegDataset(Dataset):
    def __init__(self, samples, augment: bool = True):
        self.samples = samples
        self.paired_aug = _PairedAugment(augment)
        self.augment    = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, label = self.samples[idx]

        img  = Image.open(img_path).convert("RGB")
        if mask_path and os.path.exists(mask_path):
            mask = Image.open(mask_path).convert("L")
        else:
            # Good image — all-zero mask
            mask = Image.new("L", img.size, 0)

        img, mask = self.paired_aug(img, mask)

        if self.augment:
            img = _photometric(img)

        img_t  = _to_tensor_norm(img)                           # [3, 256, 256]
        mask_t = torch.from_numpy(
            np.array(mask, dtype=np.float32) / 255.0
        ).unsqueeze(0)                                           # [1, 256, 256]
        lbl_t  = torch.tensor(label, dtype=torch.float32)

        return img_t, mask_t, lbl_t


# ──────────────────────────────────────────────────────────────────── losses ──

def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Soft Dice loss, averaged over batch."""
    pred   = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)
    inter  = (pred * target).sum(dim=1)
    return 1.0 - (2.0 * inter + eps) / (pred.sum(dim=1) + target.sum(dim=1) + eps)


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

    # Oversample defective to balance
    pos = [s for s in trn_samp if s[2] == 1]
    neg = [s for s in trn_samp if s[2] == 0]
    if len(pos) < len(neg):
        pos = pos * (len(neg) // len(pos) + 1)
        pos = pos[:len(neg)]
    trn_samp = pos + neg
    random.shuffle(trn_samp)

    train_loader = DataLoader(
        BottleSegDataset(trn_samp, augment=True),
        batch_size=args.batch, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        BottleSegDataset(val_samp, augment=False),
        batch_size=args.batch, shuffle=False, num_workers=0,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model     = ImprovedMiner(freeze_backbone=False).to(device)
    bce_cls   = nn.BCELoss()
    bce_seg   = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    alpha_cls = 0.4   # weight for classification loss
    alpha_seg = 0.6   # weight for segmentation loss

    print(f"Training improved on {len(trn_samp)} images, validating on {len(val_samp)}")
    print(f"Device: {device}  |  Epochs: {args.epochs}  |  Batch: {args.batch}")

    best_val_loss = float("inf")
    best_state    = None

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        trn_loss = trn_correct = trn_total = 0
        for imgs, masks, labels in train_loader:
            imgs, masks, labels = imgs.to(device), masks.to(device), labels.to(device)
            optimizer.zero_grad()

            conf, mask_pred = model(imgs)
            l_cls = bce_cls(conf, labels)
            l_seg = (bce_seg(mask_pred, masks) + dice_loss(mask_pred, masks).mean()) / 2

            loss = alpha_cls * l_cls + alpha_seg * l_seg
            loss.backward()
            optimizer.step()

            trn_loss    += loss.item() * len(imgs)
            trn_correct += ((conf >= 0.5).float() == labels).sum().item()
            trn_total   += len(imgs)
        scheduler.step()

        # Validate
        model.eval()
        val_loss = val_correct = val_total = 0
        with torch.no_grad():
            for imgs, masks, labels in val_loader:
                imgs, masks, labels = imgs.to(device), masks.to(device), labels.to(device)
                conf, mask_pred = model(imgs)
                l_cls = bce_cls(conf, labels)
                l_seg = (bce_seg(mask_pred, masks) + dice_loss(mask_pred, masks).mean()) / 2
                val_loss    += (alpha_cls * l_cls + alpha_seg * l_seg).item() * len(imgs)
                val_correct += ((conf >= 0.5).float() == labels).sum().item()
                val_total   += len(imgs)

        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"trn_loss={trn_loss/trn_total:.4f}  trn_acc={trn_correct/trn_total:.3f}  "
            f"val_loss={val_loss/val_total:.4f}  val_acc={val_correct/val_total:.3f}"
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
        output_names=["confidence", "mask"],
        dynamic_axes={
            "image":      {0: "batch"},
            "confidence": {0: "batch"},
            "mask":       {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )
    print(f"\nSaved ONNX model → {args.out}")


# ──────────────────────────────────────────────────────────────── entrypoint ──

def _parse():
    p = argparse.ArgumentParser(description="Train improved miner")
    p.add_argument("--data",   default="data/datasets/MVTec_AD", help="MVTec_AD root")
    p.add_argument("--out",    default="training/models/improved.onnx", help="Output ONNX path")
    p.add_argument("--epochs", type=int,   default=20)
    p.add_argument("--lr",     type=float, default=1e-4)
    p.add_argument("--batch",  type=int,   default=8)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


if __name__ == "__main__":
    train(_parse())
