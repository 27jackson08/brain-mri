"""3D U-Net definition and whole-volume inference helpers."""

from __future__ import annotations

import torch
from monai.inferers import sliding_window_inference
from monai.networks.layers import Norm
from monai.networks.nets import UNet

from common import PATCH_SIZE

MODEL_KWARGS = dict(
    spatial_dims=3,
    in_channels=4,
    out_channels=3,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.INSTANCE,
)
SW_BATCH_SIZE = 4
SW_OVERLAP = 0.5
THRESHOLD = 0.5


def build_model() -> UNet:
    return UNet(**MODEL_KWARGS)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def predict_logits(model: torch.nn.Module, image: torch.Tensor) -> torch.Tensor:
    """Sliding-window logits for a (1, 4, X, Y, Z) tensor already on the model's device."""
    return sliding_window_inference(
        inputs=image,
        roi_size=PATCH_SIZE,
        sw_batch_size=SW_BATCH_SIZE,
        predictor=model,
        overlap=SW_OVERLAP,
        mode="gaussian",
    )


def logits_to_mask(logits: torch.Tensor) -> torch.Tensor:
    """Independent sigmoid per region, thresholded to a boolean mask."""
    return torch.sigmoid(logits) > THRESHOLD
