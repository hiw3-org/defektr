"""
data/sampler.py — Block-hash seeded benchmark image sampler.

Walks a validation dataset produced by augmentations.generate_validation_dataset()
and returns a reproducible, seed-controlled sample of N images with labels and masks.

The validator derives the seed from the challenge block hash so miners cannot
predict or memorise the specific images selected each round.

Public API:
    build_ground_truth(val_dataset_path) -> list[dict]
    sample_benchmark(val_dataset_path, seed, n) -> list[dict]
    load_image_as_array(path, target_size) -> np.ndarray  [1, 3, H, W]
    load_mask_as_array(path, target_size)  -> np.ndarray  [H, W] uint8
"""

import os
import random
import hashlib
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


# ImageNet normalisation (must match training)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_DEFECT_CATEGORIES = ["synthetic", "broken_large", "broken_small", "contamination"]


# ──────────────────────────────────────────────── Image loading ───────────────

def load_image_as_array(path: str, target_size: tuple = (256, 256)) -> np.ndarray:
    """
    Load a PIL RGB image, normalise with ImageNet stats, return [1, 3, H, W] float32.
    """
    img = Image.open(path).convert("RGB").resize(target_size, Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0          # [H, W, 3]
    arr = (arr - _MEAN) / _STD
    return arr.transpose(2, 0, 1)[np.newaxis]               # [1, 3, H, W]


def load_mask_as_array(path: str, target_size: tuple = (256, 256)) -> np.ndarray:
    """
    Load a grayscale mask, binarise at 127, return [H, W] uint8 (0/1).
    """
    return (
        np.array(Image.open(path).convert("L").resize(target_size, Image.NEAREST)) > 127
    ).astype(np.uint8)


# ──────────────────────────────────────────────── Ground truth builder ────────

def build_ground_truth(val_dataset_path: str) -> list:
    """
    Walk a validation dataset directory and build the ground-truth list used
    by reward.reward():

        [{"image_id": str, "label": int (0=good/1=defective),
          "mask": np.ndarray [H,W] uint8 | None,
          "image_path": str}, ...]

    Expects the layout produced by augmentations.generate_validation_dataset():
        val_dataset_path/
            test/good/
            test/synthetic/
            test/broken_large/ broken_small/ contamination/
            ground_truth/synthetic/
            ground_truth/broken_large/ broken_small/ contamination/
    """
    records = []
    test_root = os.path.join(val_dataset_path, "test")
    gt_root   = os.path.join(val_dataset_path, "ground_truth")

    if not os.path.isdir(test_root):
        raise FileNotFoundError(f"Validation dataset not found: {val_dataset_path}")

    # Good images
    good_dir = os.path.join(test_root, "good")
    if os.path.isdir(good_dir):
        for fn in sorted(os.listdir(good_dir)):
            if _is_image(fn):
                records.append({
                    "image_id":   f"good/{fn}",
                    "label":      0,
                    "mask":       None,
                    "image_path": os.path.join(good_dir, fn),
                })

    # Defective images (synthetic + real categories)
    for cat in _DEFECT_CATEGORIES:
        cat_test_dir = os.path.join(test_root, cat)
        cat_gt_dir   = os.path.join(gt_root, cat)
        if not os.path.isdir(cat_test_dir):
            continue
        for fn in sorted(os.listdir(cat_test_dir)):
            if not _is_image(fn):
                continue
            stem     = Path(fn).stem
            mask_fn  = fn if cat == "synthetic" else f"{stem}_mask.png"
            mask_path = os.path.join(cat_gt_dir, mask_fn)
            records.append({
                "image_id":   f"{cat}/{fn}",
                "label":      1,
                "mask":       load_mask_as_array(mask_path) if os.path.isfile(mask_path) else None,
                "image_path": os.path.join(cat_test_dir, fn),
            })

    return records


# ──────────────────────────────────────────────── Seeded sampler ─────────────

def sample_benchmark(
    val_dataset_path: str,
    seed,
    n: int = 100,
) -> list:
    """
    Return a reproducible sample of N ground-truth records from the dataset.

    ``seed`` can be:
      - An integer (used directly).
      - A hex block-hash string — converted to int via SHA-256 for uniformity.
      - A bytes object.

    If n >= len(dataset), the full dataset is returned (no duplicates).
    Images are NOT loaded into memory here — callers load on demand via
    ``load_image_as_array(record["image_path"])``.

    Args:
        val_dataset_path: Path to validation dataset root.
        seed:             Block hash or integer seed.
        n:                Number of images to sample.

    Returns:
        List of ground-truth dicts (same format as build_ground_truth()).
    """
    records = build_ground_truth(val_dataset_path)
    if not records:
        raise RuntimeError(f"No images found in validation dataset: {val_dataset_path}")

    int_seed = _to_int_seed(seed)
    rng = random.Random(int_seed)

    if n >= len(records):
        rng.shuffle(records)
        return records

    return rng.sample(records, n)


# ──────────────────────────────────────────────── Helpers ────────────────────

def _is_image(filename: str) -> bool:
    return Path(filename).suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}


def _to_int_seed(seed) -> int:
    if isinstance(seed, int):
        return seed
    if isinstance(seed, (bytes, bytearray)):
        return int.from_bytes(seed[:8], "big")
    # Assume hex string (block hash)
    s = str(seed).strip().lstrip("0x")
    try:
        return int(s, 16) & 0xFFFF_FFFF_FFFF_FFFF
    except ValueError:
        # Fall back: SHA-256 of the string
        return int(hashlib.sha256(str(seed).encode()).hexdigest(), 16) & 0xFFFF_FFFF_FFFF_FFFF
