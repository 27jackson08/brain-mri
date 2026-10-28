"""Shared paths, constants, and helpers for the BraTS robustness experiment."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
DATA_ROOT = Path(os.environ.get("BRATS_DATA_ROOT", REPO_ROOT / "data"))
RAW_DIR = DATA_ROOT / "raw"
PREPROCESSED_DIR = DATA_ROOT / "preprocessed"
RUNS_DIR = Path(os.environ.get("BRATS_RUNS_DIR", DATA_ROOT / "runs"))
RESULTS_DIR = EXPERIMENT_DIR / "results"

SPLIT_FILE = PREPROCESSED_DIR / "split.json"

# Channel order of the multi-label target produced by prepare_data.py.
REGION_NAMES = ("TC", "WT", "ET")
MODALITY_NAMES = ("FLAIR", "T1w", "T1gd", "T2w")

# Training patch and the slightly larger crop that is spatially augmented before
# being centre-cropped to PATCH_SIZE, so rotations do not leave zero borders.
PATCH_SIZE = (64, 64, 64)
CROP_SIZE = (80, 80, 80)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def save_json(obj: dict | list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def case_paths(case_id: str) -> tuple[Path, Path]:
    """Return (image, label) .npy paths for a preprocessed case."""
    return (
        PREPROCESSED_DIR / f"{case_id}_image.npy",
        PREPROCESSED_DIR / f"{case_id}_label.npy",
    )
