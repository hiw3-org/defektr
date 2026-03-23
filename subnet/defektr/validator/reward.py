"""
subnet/defektr/validator/reward.py

Scoring functions for the Defektr validator.
Extracted from scripts/defektr_concept.ipynb (Nejc Rožman).

Scoring weights and thresholds come from the challenge spec JSON, not from
this module, so they can change per round without a code update.

Public API:
    classification_reward(label, confidence) -> float
    mask_iou_reward(gt_mask, pred_mask) -> float
    image_reward(label, confidence, gt_mask, pred_mask, ...) -> dict
    speed_reward(mean_latency_s, t_soft, t_hard) -> float
    robustness_reward(probe_predictions, probe_masks, mask_agree_iou) -> float
    reward(ground_truth, predictions, mean_latency_s, ...) -> float

Challenge spec scoring section (for reference):
    {
      "accuracy_weight":  0.50,   # ALPHA_ACC
      "speed_weight":     0.30,   # ALPHA_SPEED
      "robustness_weight":0.20,   # ALPHA_ROBUST
      "alpha_cls":        0.60,   # classification share within accuracy
      "alpha_loc":        0.40,   # localisation share within accuracy
      "t_soft_s":         0.200,  # seconds — full speed reward below this
      "t_hard_s":         1.000,  # seconds — zero speed reward at or above this
    }
"""

import math
from typing import Dict, List, Optional

import numpy as np


# ──────────────────────────────────────── Default weights (Phase 1) ───────────
# Used only when no challenge spec is passed to the top-level reward().

_DEFAULT_ALPHA_ACC    = 0.50
_DEFAULT_ALPHA_SPEED  = 0.30
_DEFAULT_ALPHA_ROBUST = 0.20
_DEFAULT_ALPHA_CLS    = 0.60
_DEFAULT_ALPHA_LOC    = 0.40
_DEFAULT_T_SOFT       = 0.200   # seconds
_DEFAULT_T_HARD       = 1.000   # seconds
_DEFAULT_MASK_AGREE_IOU = 0.80


# ──────────────────────────────────────── 1. Classification reward ────────────

def classification_reward(label: int, confidence: Optional[float]) -> float:
    """
    Soft reward for the binary defect/good classification head.

    Uses normalised binary cross-entropy so reward stays in [0, 1]:
        reward = 1 − BCE(label, confidence) / log(2)

    A model that outputs raw sigmoid confidence scores is rewarded for being
    calibrated (not just for crossing the 0.5 threshold).  Without a
    confidence score, maximum loss (0.0) is returned.

    Args:
        label:      Ground-truth label (1 = defective, 0 = good).
        confidence: Raw sigmoid output [0, 1], or None.

    Returns:
        float in [0, 1].
    """
    if confidence is None:
        return 0.0
    p   = float(np.clip(confidence, 1e-7, 1 - 1e-7))
    y   = float(label)
    bce = -(y * math.log(p) + (1 - y) * math.log(1 - p))
    return float(np.clip(1.0 - bce / math.log(2), 0.0, 1.0))


# ──────────────────────────────────────── 2. Mask IoU reward ─────────────────

def mask_iou_reward(
    gt_mask: Optional[np.ndarray],
    pred_mask: Optional[np.ndarray],
) -> float:
    """
    Pixel-level IoU reward for the segmentation head.

    Models without a mask head receive 0.0 — there is no partial credit for
    classification alone here (classification is handled separately).

    Special case: if the GT mask is all-zero (good image) the model is rewarded
    for returning an all-zero prediction:
        - all-zero prediction → 1.0
        - any positive pixel  → penalised proportionally

    Args:
        gt_mask:   Binary ground-truth mask (0/1 or bool) or None.
        pred_mask: Binary predicted mask (0/1 or bool) or None.

    Returns:
        float in [0, 1].  0.0 when pred_mask is None.
    """
    if pred_mask is None:
        return 0.0

    gt   = np.asarray(gt_mask,   dtype=bool) if gt_mask   is not None else np.zeros((256, 256), dtype=bool)
    pred = np.asarray(pred_mask, dtype=bool)

    intersection = np.logical_and(gt, pred).sum()
    union        = np.logical_or(gt, pred).sum()

    if union == 0:
        return 1.0  # both empty — perfect agreement on a good image

    return float(intersection) / float(union)


# ──────────────────────────────────────── 3. Per-image reward ────────────────

def image_reward(
    label: int,
    confidence: Optional[float],
    gt_mask: Optional[np.ndarray],
    pred_mask: Optional[np.ndarray],
    alpha_cls: float = _DEFAULT_ALPHA_CLS,
    alpha_loc: float = _DEFAULT_ALPHA_LOC,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Combined per-image reward.

        R_i = (α_cls × R_cls + α_loc × R_loc) / (α_cls + α_loc)

    Challenge supports both binary-only and segmentation models:
    - Binary model (no mask head): pred_mask=None → R_loc = 0, total ≤ α_cls/(α_cls+α_loc)
    - Segmentation model: pred_mask provided → can score full reward

    Args:
        label:      Ground-truth label (1 = defective, 0 = good).
        confidence: Raw sigmoid output or None.
        gt_mask:    Ground-truth binary mask (256×256) or None.
        pred_mask:  Predicted binary mask (256×256) or None.
        alpha_cls:  Weight for classification component.
        alpha_loc:  Weight for localisation component.
        verbose:    Print component scores.

    Returns:
        dict with keys 'cls', 'loc', 'total'.
    """
    cls_r = classification_reward(label, confidence)
    loc_r = mask_iou_reward(gt_mask, pred_mask)
    total = (alpha_cls * cls_r + alpha_loc * loc_r) / (alpha_cls + alpha_loc)

    result = {"cls": cls_r, "loc": loc_r, "total": total}
    if verbose:
        print(", ".join(f"{k}: {v:.3f}" for k, v in result.items()))
    return result


# ──────────────────────────────────────── 4. Speed reward ────────────────────

def speed_reward(
    mean_latency_s: float,
    t_soft: float = _DEFAULT_T_SOFT,
    t_hard: float = _DEFAULT_T_HARD,
) -> float:
    """
    Linear speed reward decaying from 1.0 to 0.0 between soft and hard thresholds.

        R_speed = max(0, 1 − (t − t_soft) / (t_hard − t_soft))

    Args:
        mean_latency_s: Mean per-image inference time in seconds (validator-measured).
        t_soft:         Below this → full speed reward (default 200ms).
        t_hard:         At or above this → zero speed reward (default 1000ms).

    Returns:
        float in [0, 1].
    """
    if mean_latency_s <= t_soft:
        return 1.0
    if mean_latency_s >= t_hard:
        return 0.0
    return 1.0 - (mean_latency_s - t_soft) / (t_hard - t_soft)


# ──────────────────────────────────────── 5. Robustness reward ───────────────

def robustness_reward(
    probe_predictions: List[List[Optional[float]]],
    probe_masks: Optional[List[List[Optional[np.ndarray]]]] = None,
    mask_agree_iou: float = _DEFAULT_MASK_AGREE_IOU,
) -> float:
    """
    Reward consistency across k augmented variants of each probe image.

    A model that has truly learned the signal will predict the same label
    regardless of the specific augmentation applied:

        R_robust = (1/m) × Σ_j  𝟙[all k predictions for probe j agree]

    For segmentation models, "agree" also requires pairwise mask IoU ≥ mask_agree_iou.

    Args:
        probe_predictions: List of m groups, each group of k sigmoid confidence
                           values (or None).  Shape: [m][k].
        probe_masks:       Optional list of m groups of k binary masks (or None).
                           Shape: [m][k].
        mask_agree_iou:    IoU threshold for mask agreement.

    Returns:
        float in [0, 1].  1.0 if no probes provided.
    """
    m = len(probe_predictions)
    if m == 0:
        return 1.0

    agreed = 0
    for j, group in enumerate(probe_predictions):
        hard_labels = [int((c or 0.0) >= 0.5) for c in group]
        label_agree = len(set(hard_labels)) == 1

        mask_agree = True
        if probe_masks is not None and j < len(probe_masks):
            masks = [mk for mk in probe_masks[j] if mk is not None]
            if len(masks) >= 2:
                pairwise = [
                    mask_iou_reward(masks[a], masks[b])
                    for a in range(len(masks))
                    for b in range(a + 1, len(masks))
                ]
                mask_agree = all(iou >= mask_agree_iou for iou in pairwise)

        if label_agree and mask_agree:
            agreed += 1

    return agreed / m


# ──────────────────────────────────────── 6. Top-level reward ────────────────

def reward(
    ground_truth: List[Dict],
    predictions: List[Dict],
    mean_latency_s: float,
    probe_predictions: Optional[List[List[Optional[float]]]] = None,
    probe_masks: Optional[List[List[Optional[np.ndarray]]]] = None,
    challenge_spec: Optional[Dict] = None,
    verbose: bool = False,
) -> float:
    """
    Score a miner's model for one validation round.

    Reads scoring weights from ``challenge_spec["scoring"]`` if provided,
    otherwise falls back to Phase 1 defaults.

    Args:
        ground_truth:      List of N dicts per image:
                             {"image_id": str, "label": int, "mask": ndarray|None}
        predictions:       List of N dicts from the model:
                             {"image_id": str, "confidence": float|None, "mask": ndarray|None}
                           Matched to ground_truth by image_id.
        mean_latency_s:    Mean per-image inference time in seconds (validator-measured).
        probe_predictions: Robustness probe confidences [m][k]. Pass None to skip.
        probe_masks:       Robustness probe masks [m][k]. Pass None to skip.
        challenge_spec:    Full challenge spec dict. Scoring section is read from
                           spec["scoring"] if present.
        verbose:           Print per-component scores.

    Returns:
        float in [0, 1].  0.0 if predictions is empty.
    """
    if not predictions:
        return 0.0

    # Read weights from challenge spec or fall back to defaults
    sc = (challenge_spec or {}).get("scoring", {})
    alpha_acc    = sc.get("accuracy_weight",   _DEFAULT_ALPHA_ACC)
    alpha_speed  = sc.get("speed_weight",      _DEFAULT_ALPHA_SPEED)
    alpha_robust = sc.get("robustness_weight", _DEFAULT_ALPHA_ROBUST)
    alpha_cls    = sc.get("alpha_cls",         _DEFAULT_ALPHA_CLS)
    alpha_loc    = sc.get("alpha_loc",         _DEFAULT_ALPHA_LOC)
    t_soft       = sc.get("t_soft_s",          _DEFAULT_T_SOFT)
    t_hard       = sc.get("t_hard_s",          _DEFAULT_T_HARD)

    pred_index = {p["image_id"]: p for p in predictions}

    per_image = []
    for gt in ground_truth:
        pred = pred_index.get(gt["image_id"], {})
        r = image_reward(
            label      = gt["label"],
            confidence = pred.get("confidence"),
            gt_mask    = gt.get("mask"),
            pred_mask  = pred.get("mask"),
            alpha_cls  = alpha_cls,
            alpha_loc  = alpha_loc,
            verbose    = verbose,
        )
        per_image.append(r["total"])

    acc_r    = float(np.mean(per_image))
    speed_r  = speed_reward(mean_latency_s, t_soft, t_hard)
    robust_r = (
        robustness_reward(probe_predictions, probe_masks)
        if probe_predictions is not None
        else 1.0
    )

    total_w = alpha_acc + alpha_speed + alpha_robust
    total_r = (alpha_acc * acc_r + alpha_speed * speed_r + alpha_robust * robust_r) / total_w

    if verbose:
        print(
            f"acc_reward={acc_r:.3f}  speed_reward={speed_r:.3f}  "
            f"robust_reward={robust_r:.3f}  total={total_r:.3f}"
        )

    return float(total_r)
