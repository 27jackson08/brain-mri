"""Training augmentation pipelines for the three experimental configurations.

The configurations form an additive ladder so that each pairwise comparison
isolates one factor:

    baseline     : foreground-biased random crops only
    spatial      : + random affine (rotation, scaling) and random flips
    degradation  : + Gaussian noise and simulated low resolution
"""

from __future__ import annotations

import math

import numpy as np
import torch
from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    EnsureTyped,
    MapTransform,
    RandAffined,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    RandSimulateLowResolutiond,
    SpatialPadd,
)

from common import CROP_SIZE, PATCH_SIZE, case_paths

KEYS = ("image", "label")

AUG_PROB = 0.5
ROTATION_RANGE_RAD = math.pi / 12  # +/- 15 degrees about each axis
SCALE_RANGE = 0.1  # isotropic scale factor in [0.9, 1.1]
NOISE_STD_MAX = 0.3  # sampled uniformly in [0, 0.3] on the normalized scale
LOWRES_ZOOM_RANGE = (0.5, 1.0)  # simulated voxel size between 1 mm and 2 mm

CONFIGS: dict[str, dict] = {
    "baseline": {
        "spatial": False,
        "degradation": False,
        "label": "Baseline",
        "description": "3D U-Net, foreground-biased random crops, no augmentation",
    },
    "spatial": {
        "spatial": True,
        "degradation": False,
        "label": "Spatial Augmentation",
        "description": "3D U-Net + random affine (rotation/scale) + random flips",
    },
    "degradation": {
        "spatial": True,
        "degradation": True,
        "label": "Degradation-Aware",
        "description": "Spatial augmentation + Gaussian noise + simulated low resolution",
    },
}


class LoadPreprocessedCased(MapTransform):
    """Load the preprocessed .npy image/label pair named by ``case_id``."""

    def __init__(self) -> None:
        super().__init__(keys=("case_id",))

    def __call__(self, data: dict) -> dict:
        d = dict(data)
        image_path, label_path = case_paths(d["case_id"])
        d["image"] = torch.from_numpy(np.load(image_path).astype(np.float32))
        d["label"] = torch.from_numpy(np.load(label_path).astype(np.float32))
        return d


def build_train_transform(config_name: str, samples_per_case: int) -> Compose:
    cfg = CONFIGS[config_name]
    # Every configuration samples the same CROP_SIZE patches (foreground = whole
    # tumour, since MONAI treats channel 0 of a multi-channel label as background
    # and takes the union of the remaining channels) and keeps their centre
    # PATCH_SIZE, so only the augmentation differs between configurations.
    steps: list = [
        LoadPreprocessedCased(),
        EnsureTyped(keys=KEYS, dtype=torch.float32, track_meta=False),
        SpatialPadd(keys=KEYS, spatial_size=CROP_SIZE),
        RandCropByPosNegLabeld(
            keys=KEYS,
            label_key="label",
            spatial_size=CROP_SIZE,
            pos=1,
            neg=1,
            num_samples=samples_per_case,
        ),
    ]
    if cfg["spatial"]:
        steps.append(
            RandAffined(
                keys=KEYS,
                prob=AUG_PROB,
                rotate_range=(ROTATION_RANGE_RAD,) * 3,
                scale_range=(SCALE_RANGE,) * 3,
                mode=("bilinear", "nearest"),
                padding_mode="zeros",
            )
        )
    steps.append(CenterSpatialCropd(keys=KEYS, roi_size=PATCH_SIZE))
    if cfg["spatial"]:
        steps.extend(RandFlipd(keys=KEYS, prob=0.5, spatial_axis=axis) for axis in range(3))
    if cfg["degradation"]:
        steps.extend(
            [
                RandGaussianNoised(keys="image", prob=AUG_PROB, mean=0.0, std=NOISE_STD_MAX, sample_std=True),
                RandSimulateLowResolutiond(
                    keys="image",
                    prob=AUG_PROB,
                    zoom_range=LOWRES_ZOOM_RANGE,
                    downsample_mode="area",
                    upsample_mode="trilinear",
                ),
            ]
        )
    return Compose(steps)


def build_eval_transform() -> Compose:
    return Compose([LoadPreprocessedCased(), EnsureTyped(keys=KEYS, dtype=torch.float32, track_meta=False)])
