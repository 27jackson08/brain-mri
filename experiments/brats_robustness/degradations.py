"""Deterministic test-time image degradations.

Both functions operate on a normalized image tensor of shape (C, X, Y, Z)
and return a new tensor; the input is never modified in place.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def add_gaussian_noise(image: torch.Tensor, sigma: float, seed: int) -> torch.Tensor:
    """Add zero-mean Gaussian noise with standard deviation ``sigma``.

    ``sigma`` is expressed in units of the normalized intensity scale, so a
    value of 0.2 corresponds to 20% of the within-brain intensity standard
    deviation of each modality. A per-case ``seed`` keeps the noise pattern
    reproducible across models being compared.
    """
    if sigma < 0:
        raise ValueError(f"sigma must be non-negative, got {sigma}")
    if sigma == 0:
        return image.clone()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(image.shape, generator=generator, dtype=torch.float32)
    return image.float() + sigma * noise.to(image.device)


def reduce_resolution(image: torch.Tensor, factor: float) -> torch.Tensor:
    """Simulate a lower acquisition resolution.

    The volume is downsampled by ``factor`` with area (box) interpolation,
    which models partial-volume averaging, then upsampled back to the
    original grid with trilinear interpolation so that predictions can be
    compared against the original 1 mm reference labels.
    """
    if not 0 < factor <= 1:
        raise ValueError(f"factor must be in (0, 1], got {factor}")
    if factor == 1:
        return image.clone()
    original_shape = image.shape[1:]
    low_shape = [max(1, int(round(s * factor))) for s in original_shape]
    batched = image.float().unsqueeze(0)
    low = F.interpolate(batched, size=low_shape, mode="area")
    restored = F.interpolate(low, size=original_shape, mode="trilinear", align_corners=False)
    return restored.squeeze(0)


def apply_degradation(image: torch.Tensor, kind: str, level: float, seed: int) -> torch.Tensor:
    """Dispatch on ``kind`` in {"clean", "noise", "lowres"}."""
    if kind == "clean":
        return image.clone()
    if kind == "noise":
        return add_gaussian_noise(image, sigma=level, seed=seed)
    if kind == "lowres":
        return reduce_resolution(image, factor=level)
    raise ValueError(f"unknown degradation kind: {kind}")
