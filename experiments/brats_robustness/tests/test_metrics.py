import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metrics import dice_per_region, hd95_per_region, summarize  # noqa: E402


def _cube(size, lo, hi):
    mask = torch.zeros(1, size, size, size, dtype=torch.bool)
    mask[:, lo:hi, lo:hi, lo:hi] = True
    return mask


def test_dice_is_one_for_identical_masks():
    mask = _cube(16, 4, 12)
    assert dice_per_region(mask, mask) == [1.0]


def test_dice_follows_brats_convention_for_empty_masks():
    empty = torch.zeros(2, 8, 8, 8, dtype=torch.bool)
    half_empty = empty.clone()
    half_empty[1, 2:4, 2:4, 2:4] = True
    assert dice_per_region(empty, empty) == [1.0, 1.0]
    assert dice_per_region(half_empty, empty) == [1.0, 0.0]


def test_dice_partial_overlap():
    pred = _cube(16, 0, 8)  # 512 voxels
    target = _cube(16, 4, 12)  # 512 voxels, overlap 4^3 = 64
    assert dice_per_region(pred, target) == pytest.approx([2 * 64 / 1024])


def test_dice_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        dice_per_region(_cube(8, 0, 4), _cube(16, 0, 4))


def test_hd95_zero_for_identical_and_nan_for_empty():
    mask = _cube(16, 4, 12)
    assert hd95_per_region(mask, mask) == [0.0]
    empty = torch.zeros_like(mask)
    assert math.isnan(hd95_per_region(mask, empty)[0])


def test_hd95_detects_shift():
    a = _cube(24, 4, 12)
    b = _cube(24, 8, 16)
    (hd,) = hd95_per_region(a, b)
    assert 2.0 < hd <= 8.0


def test_summarize_ignores_nan():
    s = summarize([1.0, float("nan"), 3.0])
    assert s == {"mean": 2.0, "std": 1.0, "n": 2}
    assert summarize([float("nan")])["n"] == 0
