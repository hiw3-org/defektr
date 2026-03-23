"""
subnet/defektr/validator/augmentations.py

Augmentation pipeline and synthetic defect generation.
Extracted from scripts/defektr_concept.ipynb (Nejc Rožman).

Public API:
    generate_defect_mask(height, width, rng, center=None) -> np.ndarray
    inject_defect(image, texture_dirs, rng) -> (PIL.Image, PIL.Image)
    augment(image, rng, mask=None, target_size=(256,256)) -> (PIL.Image, PIL.Image|None)
    generate_validation_dataset(src_dataset_path, output_path, dtd_path, ...)
"""

import os
import random
import shutil

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import opensimplex


# ──────────────────────────────────────────────────── Defect mask ─────────────

def generate_defect_mask(
    height: int,
    width: int,
    rng: random.Random,
    center: tuple = None,
) -> np.ndarray:
    """
    Generate a single connected binary defect mask using Perlin noise.

    Multi-octave Perlin noise is evaluated on a grid centered at ``center``,
    then Gaussian-weighted so only one coherent blob region exceeds the
    threshold. The result is an organic, fractal-edged defect region that
    closely resembles real anomaly shapes (scratches, contamination patches).

    Args:
        height: Mask height in pixels.
        width:  Mask width in pixels.
        rng:    Seeded random instance for reproducibility.
        center: (cy, cx) pixel coordinates for the defect center.
                Defaults to a random point in the central 60% of the image.

    Returns:
        float32 ndarray (height, width) with values in {0.0, 1.0}.
    """
    if center is None:
        cy = rng.uniform(0.2, 0.8) * height
        cx = rng.uniform(0.2, 0.8) * width
    else:
        cy, cx = float(center[0]), float(center[1])

    scale       = rng.uniform(2.0, 6.0)
    octaves     = rng.randint(4, 8)
    persistence = rng.uniform(0.4, 0.7)
    lacunarity  = rng.uniform(1.8, 2.2)
    base        = rng.randint(0, 255)

    opensimplex.seed(base)
    # Build grid and evaluate simplex noise (vectorised)
    xs = np.linspace(0, scale, width,  dtype=np.float32)
    ys = np.linspace(0, scale, height, dtype=np.float32)
    perlin = np.zeros((height, width), dtype=np.float32)
    for o in range(octaves):
        freq = lacunarity ** o
        amp  = persistence ** o
        for yi in range(height):
            for xi in range(width):
                perlin[yi, xi] += amp * opensimplex.noise2(xs[xi] * freq, ys[yi] * freq)


    perlin = (perlin - perlin.min()) / (perlin.max() - perlin.min() + 1e-8)

    sigma  = rng.uniform(0.06, 0.09) * min(height, width)
    yy, xx = np.mgrid[0:height, 0:width]
    weight = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))

    combined = perlin * weight
    target_coverage = rng.uniform(0.0001, 0.02)
    threshold = float(np.percentile(combined, (1.0 - target_coverage) * 100))
    return (combined >= threshold).astype(np.float32)


# ──────────────────────────────────────────────────── Defect injection ────────

def inject_defect(
    image: Image.Image,
    texture_dirs: list,
    rng: random.Random,
) -> tuple:
    """
    Inject a synthetic defect into ``image`` using DTD texture blending.

    The defect center is sampled from foreground pixels only. A random texture
    patch is blended in within the Perlin-noise mask:
        result = image × (1 − mask) + texture × mask

    Args:
        image:        PIL RGB image to corrupt.
        texture_dirs: Directories from which to sample texture images
                      (typically ``train/good`` folders of other MVTec categories).
        rng:          Seeded random instance.

    Returns:
        (defected_image, mask_image) — PIL RGB + grayscale PIL (mode "L", 0/255).
    """
    w, h = image.size
    img_arr = np.array(image, dtype=np.float32) / 255.0

    cs = max(5, min(h, w) // 20)
    corner_pixels = np.concatenate([
        img_arr[:cs, :cs].reshape(-1, 3),
        img_arr[:cs, -cs:].reshape(-1, 3),
        img_arr[-cs:, :cs].reshape(-1, 3),
        img_arr[-cs:, -cs:].reshape(-1, 3),
    ], axis=0)
    bg_color = corner_pixels.mean(axis=0)
    fg_mask  = (np.linalg.norm(img_arr - bg_color, axis=2) > 0.1).astype(np.float32)

    fg_pixels = np.argwhere(fg_mask > 0)
    if len(fg_pixels) > 0:
        cy, cx = fg_pixels[rng.randint(0, len(fg_pixels) - 1)]
    else:
        cy, cx = h // 2, w // 2

    texture_dir   = rng.choice(texture_dirs)
    texture_files = sorted(
        f for f in os.listdir(texture_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
    )
    if not texture_files:
        raise ValueError(f"No texture images found in: {texture_dir}")

    texture_arr = np.array(
        Image.open(os.path.join(texture_dir, rng.choice(texture_files)))
        .convert("RGB").resize((w, h), Image.BILINEAR),
        dtype=np.float32,
    ) / 255.0

    mask    = generate_defect_mask(h, w, rng, center=(cy, cx)) * fg_mask
    mask_3d = mask[:, :, np.newaxis]
    defected = np.clip((img_arr * (1.0 - mask_3d) + texture_arr * mask_3d) * 255, 0, 255).astype(np.uint8)

    return Image.fromarray(defected), Image.fromarray((mask * 255).astype(np.uint8))


# ──────────────────────────────────────────────────── Augmentation ────────────

def augment(
    image: Image.Image,
    rng: random.Random,
    mask: Image.Image = None,
    target_size: tuple = (256, 256),
) -> tuple:
    """
    Reproducible geometric + photometric augmentation pipeline.

    Geometric transforms (flip, rotation, crop) are applied identically to
    image and mask.  Photometric transforms (noise, brightness, contrast, blur)
    are image-only.  Serves two purposes:
      - Anti-lookup: makes each round's images unique vs. known MVTec ground truth.
      - Edge simulation: mimics imaging conditions on RPi / Jetson cameras.

    Args:
        image:       PIL RGB image.
        rng:         Seeded random instance.
        mask:        Optional grayscale PIL mask (mode "L") — geometric transforms
                     are mirrored using nearest-neighbour resampling.
        target_size: (width, height) of output. Default (256, 256).

    Returns:
        (aug_image, aug_mask) — aug_mask is None when no mask was provided.
    """
    # ── Geometric ────────────────────────────────────────────────────────────
    if rng.random() < 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        if mask is not None:
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

    if rng.random() < 0.5:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        if mask is not None:
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

    angle = rng.uniform(-15, 15)
    image = image.rotate(angle, resample=Image.BILINEAR,  expand=False)
    if mask is not None:
        mask = mask.rotate(angle,  resample=Image.NEAREST, expand=False)

    w, h = image.size
    crop_scale = rng.uniform(0.8, 1.0)
    cw, ch = int(w * crop_scale), int(h * crop_scale)
    x0 = rng.randint(0, w - cw)
    y0 = rng.randint(0, h - ch)
    box = (x0, y0, x0 + cw, y0 + ch)
    image = image.crop(box).resize((w, h), Image.BILINEAR)
    if mask is not None:
        mask = mask.crop(box).resize((w, h), Image.NEAREST)

    image = image.resize(target_size, Image.BILINEAR)
    if mask is not None:
        mask = mask.resize(target_size, Image.NEAREST)

    # ── Photometric ──────────────────────────────────────────────────────────
    noise_sigma = rng.uniform(0.0, 8.0) / 255.0
    if noise_sigma > 0:
        np_rng = np.random.RandomState(rng.randint(0, 2**31 - 1))
        arr = np.array(image, dtype=np.float32) / 255.0
        arr += np_rng.randn(*arr.shape).astype(np.float32) * noise_sigma
        image = Image.fromarray(np.clip(arr * 255, 0, 255).astype(np.uint8))

    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.85, 1.15))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.85, 1.15))

    if rng.random() < 0.5:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.0, 1.5)))

    return image, mask


# ──────────────────────────────────────────────── Dataset builder ─────────────

def generate_validation_dataset(
    src_dataset_path: str,
    output_path: str,
    dtd_path: str,
    num_images: int = 20,
    seed: int = 42,
    p: float = 0.5,
    real_defect_ratio: float = 0.3,
) -> None:
    """
    Generate a hybrid validation dataset from MVTec_AD bottle category.

    ``num_images`` images total, split between good (prob 1-p) and defective
    (prob p).  Defective images are ``real_defect_ratio`` real MVTec samples +
    remainder synthetic (Perlin mask + DTD texture blend).  Augmentation is
    applied to all images.

    Output layout:
        output_path/
            test/good/
            test/synthetic/
            test/broken_large/ broken_small/ contamination/
            ground_truth/synthetic/
            ground_truth/broken_large/ broken_small/ contamination/

    Args:
        src_dataset_path: Path to MVTec_AD root (contains ``bottle/``).
        output_path:      Destination folder (wiped and recreated each call).
        dtd_path:         Path to DTD dataset root (contains ``images/``).
        num_images:       Total images in output dataset.
        seed:             Random seed — use block hash each round.
        p:                Probability an image is defective.
        real_defect_ratio: Fraction of defective images from real MVTec test split.
    """
    good_train_dir = os.path.join(src_dataset_path, "bottle", "train", "good")
    test_dir       = os.path.join(src_dataset_path, "bottle", "test")
    gt_dir         = os.path.join(src_dataset_path, "bottle", "ground_truth")
    real_cats      = ["broken_large", "broken_small", "contamination"]

    for d in [good_train_dir, test_dir, gt_dir]:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Required directory not found: {d}")

    dtd_images_dir = os.path.join(dtd_path, "images")
    if not os.path.isdir(dtd_images_dir):
        raise FileNotFoundError(f"DTD images directory not found: {dtd_images_dir}")

    texture_dirs = sorted(
        os.path.join(dtd_images_dir, cat)
        for cat in os.listdir(dtd_images_dir)
        if os.path.isdir(os.path.join(dtd_images_dir, cat))
    )
    if not texture_dirs:
        raise RuntimeError(f"No texture categories found in: {dtd_images_dir}")

    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    rng           = random.Random(seed)
    num_defective = round(num_images * p)
    num_good      = num_images - num_defective
    num_real      = round(num_defective * real_defect_ratio)
    num_synthetic = num_defective - num_real

    def _mkdir(*parts):
        path = os.path.join(output_path, *parts)
        os.makedirs(path, exist_ok=True)
        return path

    test_good_dir = _mkdir("test", "good")
    test_syn_dir  = _mkdir("test", "synthetic")
    gt_syn_dir    = _mkdir("ground_truth", "synthetic")
    real_out_dirs = {
        cat: (_mkdir("test", cat), _mkdir("ground_truth", cat))
        for cat in real_cats
    }

    train_good_files = sorted(
        f for f in os.listdir(good_train_dir)
        if os.path.isfile(os.path.join(good_train_dir, f))
    )
    selected_good = rng.sample(train_good_files, min(num_good, len(train_good_files)))
    for fn in selected_good:
        img, _ = augment(Image.open(os.path.join(good_train_dir, fn)).convert("RGB"), rng)
        img.save(os.path.join(test_good_dir, fn))

    synthetic_pool   = [f for f in train_good_files if f not in selected_good]
    selected_synth   = rng.sample(synthetic_pool, min(num_synthetic, len(synthetic_pool)))
    for fn in selected_synth:
        img, msk = inject_defect(Image.open(os.path.join(good_train_dir, fn)).convert("RGB"), texture_dirs, rng)
        img, msk = augment(img, rng, mask=msk)
        img.save(os.path.join(test_syn_dir, fn))
        msk.save(os.path.join(gt_syn_dir, fn))

    real_pool    = [(cat, f) for cat in real_cats
                   for f in sorted(os.listdir(os.path.join(test_dir, cat)))
                   if os.path.isfile(os.path.join(test_dir, cat, f))]
    selected_real = rng.sample(real_pool, min(num_real, len(real_pool)))
    for cat, fn in selected_real:
        stem = os.path.splitext(fn)[0]
        img  = Image.open(os.path.join(test_dir, cat, fn)).convert("RGB")
        msk  = Image.open(os.path.join(gt_dir, cat, f"{stem}_mask.png")).convert("L")
        img, msk = augment(img, rng, mask=msk)
        tdir, gdir = real_out_dirs[cat]
        img.save(os.path.join(tdir, fn))
        msk.save(os.path.join(gdir, f"{stem}_mask.png"))

    print(
        f"Validation dataset created at '{output_path}':\n"
        f"  {num_good} good images\n"
        f"  {len(selected_synth)} synthetic defective images\n"
        f"  {len(selected_real)} real defective images\n"
        f"  seed={seed}"
    )
