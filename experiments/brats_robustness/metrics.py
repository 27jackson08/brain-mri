"""Segmentation metrics computed per case on binary multi-label masks."""

from __future__ import annotations

import math

import numpy as np
import torch
from monai.metrics import compute_hausdorff_distance


def dice_per_region(pred: torch.Tensor, target: torch.Tensor) -> list[float]:
    """Dice per channel for (C, X, Y, Z) binary tensors.

    Follows the BraTS convention: if both the prediction and the reference
    are empty for a region the Dice is 1.0; if only one is empty it is 0.0.
    """
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
    pred_b = pred.bool().reshape(pred.shape[0], -1)
    target_b = target.bool().reshape(target.shape[0], -1)
    # Integer sums, then Python floats: MPS tensors cannot be cast to float64.
    intersection = (pred_b & target_b).sum(dim=1).tolist()
    denominator = (pred_b.sum(dim=1) + target_b.sum(dim=1)).tolist()
    return [1.0 if denom == 0 else 2.0 * inter / denom for inter, denom in zip(intersection, denominator)]


def hd95_per_region(pred: torch.Tensor, target: torch.Tensor) -> list[float]:
    """95th-percentile Hausdorff distance (voxels) per channel.

    Returns NaN for a region when either mask is empty, since the distance
    is undefined; callers should report how many cases were excluded.
    """
    pred_np = pred.bool().cpu().numpy()[None]
    target_np = target.bool().cpu().numpy()[None]
    out: list[float] = []
    for c in range(pred_np.shape[1]):
        p = pred_np[:, c : c + 1]
        t = target_np[:, c : c + 1]
        if p.sum() == 0 or t.sum() == 0:
            out.append(math.nan)
            continue
        hd = compute_hausdorff_distance(
            torch.from_numpy(p), torch.from_numpy(t), include_background=True, percentile=95
        )
        out.append(float(hd.item()))
    return out


def summarize(values: list[float]) -> dict[str, float]:
    """Mean, standard deviation, and count ignoring NaNs."""
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return {"mean": math.nan, "std": math.nan, "n": 0}
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0)), "n": int(arr.size)}
