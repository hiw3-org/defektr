"""
scripts/visualize_models.py

Runs the baseline and improved ONNX models on a sample defective image
and saves a side-by-side comparison PNG.

Usage:
    python scripts/visualize_models.py
    python scripts/visualize_models.py --image data/datasets/bottle/test/broken_large/000.png
    python scripts/visualize_models.py --out output.png
"""

import sys
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import onnxruntime as ort
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
from data.sampler import load_image_as_array

_BASELINE = _ROOT / "training" / "models" / "baseline.onnx"
_IMPROVED = _ROOT / "training" / "models" / "improved.onnx"
_DEFAULT_IMAGE = _ROOT / "data" / "datasets" / "bottle" / "test" / "broken_large" / "000.png"
_THRESHOLD = 0.5


def run_model(model_path: str, image_array: np.ndarray):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    session = ort.InferenceSession(str(model_path), sess_options=opts,
                                   providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: image_array})
    confidence = float(outputs[0].flatten()[0])
    mask = None
    if len(outputs) > 1:
        raw = outputs[1].squeeze()
        mask = (raw >= _THRESHOLD).astype(np.uint8)
    return confidence, mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=str(_DEFAULT_IMAGE))
    parser.add_argument("--out", default="model_comparison.png")
    args = parser.parse_args()

    image_path = args.image
    img_pil = Image.open(image_path).convert("RGB").resize((256, 256), Image.BILINEAR)
    img_np = np.array(img_pil)
    img_array = load_image_as_array(image_path)

    baseline_conf, baseline_mask = run_model(_BASELINE, img_array)
    improved_conf, improved_mask = run_model(_IMPROVED, img_array)

    baseline_label = "DEFECT" if baseline_conf >= _THRESHOLD else "OK"
    improved_label = "DEFECT" if improved_conf >= _THRESHOLD else "OK"

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#1a1a2e")
    for ax in axes:
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    # Panel 1 — original image
    axes[0].imshow(img_np)
    axes[0].set_title("Input Image", color="white", fontsize=14, pad=12)
    axes[0].axis("off")

    # Panel 2 — baseline output
    axes[1].imshow(img_np)
    color = "#e74c3c" if baseline_label == "DEFECT" else "#2ecc71"
    axes[1].set_title(
        f"Miner 1 — Baseline (MobileNetV2)\nConfidence: {baseline_conf:.3f}  →  {baseline_label}",
        color="white", fontsize=13, pad=12
    )
    axes[1].text(
        0.5, 0.05, baseline_label,
        transform=axes[1].transAxes,
        ha="center", va="bottom",
        fontsize=20, fontweight="bold", color=color,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a2e", edgecolor=color, linewidth=2)
    )
    axes[1].axis("off")

    # Panel 3 — improved output with mask overlay
    axes[2].imshow(img_np)
    if improved_mask is not None:
        overlay = np.zeros((*improved_mask.shape, 4), dtype=np.float32)
        overlay[improved_mask == 1] = [1.0, 0.2, 0.2, 0.5]  # red, semi-transparent
        axes[2].imshow(overlay)
        patch = mpatches.Patch(color=(1.0, 0.2, 0.2, 0.7), label="Defect region")
        axes[2].legend(handles=[patch], loc="lower right",
                       facecolor="#1a1a2e", edgecolor="#666", labelcolor="white", fontsize=10)
    color = "#e74c3c" if improved_label == "DEFECT" else "#2ecc71"
    axes[2].set_title(
        f"Miner 2 — Improved (MobileNetV2 + U-Net)\nConfidence: {improved_conf:.3f}  →  {improved_label}  +  segmentation mask",
        color="white", fontsize=13, pad=12
    )
    axes[2].text(
        0.5, 0.05, improved_label,
        transform=axes[2].transAxes,
        ha="center", va="bottom",
        fontsize=20, fontweight="bold", color=color,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a2e", edgecolor=color, linewidth=2)
    )
    axes[2].axis("off")

    plt.suptitle("Defektr — Model Output Comparison", color="white", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved → {args.out}")
    print(f"  Baseline : confidence={baseline_conf:.4f}  label={baseline_label}  mask=no")
    print(f"  Improved : confidence={improved_conf:.4f}  label={improved_label}  mask={'yes' if improved_mask is not None else 'no'}")


if __name__ == "__main__":
    main()
