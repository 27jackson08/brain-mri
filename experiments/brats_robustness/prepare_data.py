"""Extract MSD Task01 BrainTumour, create a fixed split, and preprocess cases.

Each selected case is reoriented to RAS, cropped to the brain bounding box,
z-score normalized per modality over non-zero voxels, and its label map is
converted to the three nested BraTS regions (TC, WT, ET) following the label
definitions in the dataset's own ``dataset.json``:

    1 = edema, 2 = non-enhancing tumor, 3 = enhancing tumour

Outputs ``<case>_image.npy`` (4, X, Y, Z) float16 and ``<case>_label.npy``
(3, X, Y, Z) uint8 under the preprocessed directory, plus ``split.json``.
"""

from __future__ import annotations

import argparse
import random
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np

from common import PREPROCESSED_DIR, RAW_DIR, SPLIT_FILE, case_paths, load_json, save_json

TASK_NAME = "Task01_BrainTumour"
EDEMA, NON_ENHANCING, ENHANCING = 1, 2, 3


def convert_labels(label: np.ndarray) -> np.ndarray:
    """Map an integer label volume to a (3, ...) uint8 array of TC, WT, ET."""
    tumor_core = (label == NON_ENHANCING) | (label == ENHANCING)
    whole_tumor = tumor_core | (label == EDEMA)
    enhancing = label == ENHANCING
    return np.stack([tumor_core, whole_tumor, enhancing], axis=0).astype(np.uint8)


def normalize_nonzero(image: np.ndarray) -> np.ndarray:
    """Per-channel z-score over non-zero voxels; zeros stay zero."""
    out = np.zeros_like(image, dtype=np.float32)
    for c in range(image.shape[0]):
        channel = image[c]
        mask = channel != 0
        if mask.sum() < 2:
            continue
        values = channel[mask]
        std = values.std()
        if std <= 0:
            continue
        out[c][mask] = (values - values.mean()) / std
    return out


def foreground_bbox(mask: np.ndarray) -> tuple[slice, ...]:
    """Tight bounding box around the True voxels of a 3-D mask."""
    if not mask.any():
        raise ValueError("empty foreground mask")
    slices = []
    for axis in range(mask.ndim):
        other = tuple(a for a in range(mask.ndim) if a != axis)
        nonzero = np.where(mask.any(axis=other))[0]
        slices.append(slice(int(nonzero[0]), int(nonzero[-1]) + 1))
    return tuple(slices)


def extract_if_needed(raw_dir: Path) -> Path:
    task_dir = raw_dir / TASK_NAME
    if (task_dir / "dataset.json").exists() and (task_dir / "labelsTr").is_dir():
        return task_dir
    tar_path = raw_dir / f"{TASK_NAME}.tar"
    if not tar_path.exists():
        raise FileNotFoundError(f"expected {tar_path}; download it from the MSD mirror first")
    print(f"extracting {tar_path} ...", flush=True)
    with tarfile.open(tar_path) as tar:
        tar.extractall(raw_dir, filter="data")
    return task_dir


def preprocess_case(job: tuple[str, str, str]) -> dict:
    case_id, image_path, label_path = job
    image_nii = nib.as_closest_canonical(nib.load(image_path))
    label_nii = nib.as_closest_canonical(nib.load(label_path))
    image = np.asarray(image_nii.dataobj, dtype=np.float32)  # (X, Y, Z, 4)
    label = np.asarray(label_nii.dataobj).astype(np.uint8)  # (X, Y, Z)
    image = np.moveaxis(image, -1, 0)  # (4, X, Y, Z)

    bbox = foreground_bbox((image != 0).any(axis=0))
    image = image[(slice(None),) + bbox]
    label = label[bbox]
    image = normalize_nonzero(image)
    regions = convert_labels(label)

    img_out, lab_out = case_paths(case_id)
    np.save(img_out, image.astype(np.float16))
    np.save(lab_out, regions)

    t1gd = image[2]
    return {
        "case_id": case_id,
        "shape": list(image.shape[1:]),
        "bbox": [[s.start, s.stop] for s in bbox],
        "spacing": [float(z) for z in image_nii.header.get_zooms()[:3]],
        "voxels": {"TC": int(regions[0].sum()), "WT": int(regions[1].sum()), "ET": int(regions[2].sum())},
        "t1gd_mean_non_enhancing": float(t1gd[label == NON_ENHANCING].mean()) if (label == NON_ENHANCING).any() else None,
        "t1gd_mean_enhancing": float(t1gd[label == ENHANCING].mean()) if (label == ENHANCING).any() else None,
    }


def make_split(case_ids: list[str], n_train: int, n_val: int, n_test: int, seed: int) -> dict:
    ids = sorted(case_ids)
    random.Random(seed).shuffle(ids)
    needed = n_train + n_val + n_test
    if needed > len(ids):
        raise ValueError(f"requested {needed} cases but only {len(ids)} available")
    return {
        "seed": seed,
        "train": ids[:n_train],
        "val": ids[n_train : n_train + n_val],
        "test": ids[n_train + n_val : needed],
        "unused": ids[needed:],
    }


def report_label_sanity(metas: list[dict]) -> dict:
    enh = [m["t1gd_mean_enhancing"] for m in metas if m["t1gd_mean_enhancing"] is not None]
    non = [m["t1gd_mean_non_enhancing"] for m in metas if m["t1gd_mean_non_enhancing"] is not None]
    summary = {
        "n_cases": len(metas),
        "n_cases_without_ET": sum(1 for m in metas if m["voxels"]["ET"] == 0),
        "t1gd_mean_enhancing": float(np.mean(enh)) if enh else None,
        "t1gd_mean_non_enhancing": float(np.mean(non)) if non else None,
    }
    def show(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.3f}"

    print(
        "label sanity check: mean normalized T1gd intensity "
        f"enhancing={show(summary['t1gd_mean_enhancing'])} vs non-enhancing={show(summary['t1gd_mean_non_enhancing'])} "
        f"(enhancing tumour should be brighter); cases without ET: {summary['n_cases_without_ET']}/{len(metas)}",
        flush=True,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-train", type=int, default=100)
    parser.add_argument("--n-val", type=int, default=15)
    parser.add_argument("--n-test", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    task_dir = extract_if_needed(RAW_DIR)
    dataset = load_json(task_dir / "dataset.json")
    entries = {Path(e["image"]).name.replace(".nii.gz", ""): e for e in dataset["training"]}
    split = make_split(list(entries), args.n_train, args.n_val, args.n_test, args.seed)
    split["labels"] = dataset["labels"]
    split["modality"] = dataset["modality"]
    PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    save_json(split, SPLIT_FILE)

    selected = split["train"] + split["val"] + split["test"]
    jobs = []
    for case_id in selected:
        img_out, lab_out = case_paths(case_id)
        if img_out.exists() and lab_out.exists():
            continue
        e = entries[case_id]
        jobs.append((case_id, str(task_dir / e["image"]), str(task_dir / e["label"])))
    print(f"split: {len(split['train'])} train / {len(split['val'])} val / {len(split['test'])} test; "
          f"{len(jobs)} cases to preprocess", flush=True)

    start = time.time()
    metas: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, meta in enumerate(pool.map(preprocess_case, jobs), 1):
            metas.append(meta)
            if i % 10 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} done ({time.time() - start:.0f}s)", flush=True)

    meta_file = PREPROCESSED_DIR / "cases_meta.json"
    existing = load_json(meta_file) if meta_file.exists() else {}
    merged = {**existing, **{m["case_id"]: m for m in metas}}
    save_json(merged, meta_file)
    sanity = report_label_sanity(list(merged.values()))
    save_json(sanity, PREPROCESSED_DIR / "label_sanity.json")


if __name__ == "__main__":
    main()
