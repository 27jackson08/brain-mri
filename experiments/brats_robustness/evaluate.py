"""Evaluate trained models on the held-out test split under image degradations.

For every (model, test case, condition) the script records per-region Dice,
95th-percentile Hausdorff distance, and inference time. Degradations are
applied deterministically with a per-case seed so that every model sees the
exact same corrupted input, which allows paired comparisons.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from common import REGION_NAMES, RESULTS_DIR, RUNS_DIR, SPLIT_FILE, get_device, load_json, save_json
from degradations import apply_degradation
from metrics import dice_per_region, hd95_per_region, summarize
from model import build_model, logits_to_mask, predict_logits
from transforms import LoadPreprocessedCased

# (kind, level). Noise sigma is on the normalized intensity scale; lowres
# level is the downsampling factor (0.5 ~ 2 mm voxels, 0.333 ~ 3 mm voxels).
DEFAULT_CONDITIONS: tuple[tuple[str, float], ...] = (
    ("clean", 1.0),
    ("noise", 0.1),
    ("noise", 0.2),
    ("noise", 0.4),
    ("lowres", 0.5),
    ("lowres", 0.333),
)
NOISE_SEED_BASE = 20261028
CSV_FIELDS = ["run", "config", "seed", "case_id", "condition", "level", *[f"dice_{r}" for r in REGION_NAMES],
              "dice_mean", *[f"hd95_{r}" for r in REGION_NAMES], *[f"gt_voxels_{r}" for r in REGION_NAMES],
              *[f"pred_voxels_{r}" for r in REGION_NAMES], "infer_time_s"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", default=["baseline_seed0", "spatial_seed0", "degradation_seed0"])
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--no-hd95", action="store_true")
    parser.add_argument("--vis-case-index", type=int, default=0, help="test case whose predictions are saved for figures")
    parser.add_argument("--out", default=str(RESULTS_DIR / "per_case_results.csv"))
    return parser.parse_args()


def parse_condition(kind: str, level: float) -> str:
    return "clean" if kind == "clean" else f"{kind}_{level:g}"


def load_model(run_dir: Path, device: torch.device) -> torch.nn.Module:
    model = build_model()
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True))
    return model.to(device).eval()


def evaluate_case(model, image: torch.Tensor, label: torch.Tensor, device, conditions, case_seed: int,
                  with_hd95: bool, keep_predictions: bool) -> tuple[list[dict], dict[str, np.ndarray]]:
    rows, predictions = [], {}
    label_dev = label.to(device)
    gt_voxels = label_dev.sum(dim=(1, 2, 3)).tolist()
    for kind, level in conditions:
        degraded = apply_degradation(image, kind, level, seed=case_seed)
        if device.type == "mps":
            torch.mps.synchronize()
        start = time.time()
        pred = logits_to_mask(predict_logits(model, degraded.unsqueeze(0).to(device)))[0]
        if device.type == "mps":
            torch.mps.synchronize()
        infer_time = time.time() - start
        dice = dice_per_region(pred, label_dev)
        hd95 = hd95_per_region(pred.cpu(), label) if with_hd95 else [float("nan")] * len(REGION_NAMES)
        pred_voxels = pred.sum(dim=(1, 2, 3)).tolist()
        rows.append({"condition": parse_condition(kind, level), "level": level,
                     **{f"dice_{r}": d for r, d in zip(REGION_NAMES, dice)}, "dice_mean": float(np.mean(dice)),
                     **{f"hd95_{r}": h for r, h in zip(REGION_NAMES, hd95)},
                     **{f"gt_voxels_{r}": int(v) for r, v in zip(REGION_NAMES, gt_voxels)},
                     **{f"pred_voxels_{r}": int(v) for r, v in zip(REGION_NAMES, pred_voxels)},
                     "infer_time_s": infer_time})
        if keep_predictions:
            predictions[parse_condition(kind, level)] = pred.cpu().numpy().astype(np.uint8)
    return rows, predictions


def aggregate(rows: list[dict]) -> dict:
    summary: dict = {}
    for row in rows:
        bucket = summary.setdefault(row["run"], {}).setdefault(row["condition"], {})
        for key in [*[f"dice_{r}" for r in REGION_NAMES], "dice_mean", *[f"hd95_{r}" for r in REGION_NAMES],
                    "infer_time_s"]:
            bucket.setdefault(key, []).append(float(row[key]))
    return {run: {cond: {k: summarize(v) for k, v in metrics.items()} for cond, metrics in conds.items()}
            for run, conds in summary.items()}


def main() -> None:
    args = parse_args()
    device = get_device()
    split = load_json(SPLIT_FILE)
    case_ids = split[args.split][: args.max_cases]
    if not 0 <= args.vis_case_index < len(case_ids):
        raise SystemExit(f"--vis-case-index must be in [0, {len(case_ids) - 1}]")
    loader = LoadPreprocessedCased()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".partial.csv")
    vis_dir = RUNS_DIR / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)
    print(f"evaluating {len(args.runs)} runs x {len(case_ids)} {args.split} cases x {len(DEFAULT_CONDITIONS)} "
          f"conditions on {device}", flush=True)

    all_rows: list[dict] = []
    with open(tmp_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for run in args.runs:
            run_dir = RUNS_DIR / run
            run_cfg = load_json(run_dir / "config.json")
            model = load_model(run_dir, device)
            # Untimed warm-up so Metal kernel compilation is not charged to the first case.
            predict_logits(model, loader({"case_id": case_ids[0]})["image"].unsqueeze(0).to(device))
            start = time.time()
            for idx, case_id in enumerate(case_ids):
                data = loader({"case_id": case_id})
                rows, preds = evaluate_case(model, data["image"], data["label"], device, DEFAULT_CONDITIONS,
                                            case_seed=NOISE_SEED_BASE + idx, with_hd95=not args.no_hd95,
                                            keep_predictions=idx == args.vis_case_index)
                for row in rows:
                    full = {"run": run, "config": run_cfg["config"], "seed": run_cfg["seed"], "case_id": case_id, **row}
                    writer.writerow(full)
                    all_rows.append(full)
                f.flush()
                for cond, arr in preds.items():
                    np.save(vis_dir / f"{run}__{case_id}__{cond}.npy", arr)
                clean = next(r for r in rows if r["condition"] == "clean")
                print(f"  [{run}] {idx + 1}/{len(case_ids)} {case_id} clean Dice mean={clean['dice_mean']:.3f} "
                      f"({time.time() - start:.0f}s)", flush=True)
            del model
            if device.type == "mps":
                torch.mps.empty_cache()

    tmp_path.replace(out_path)
    summary = aggregate(all_rows)
    save_json({"split": args.split, "n_cases": len(case_ids), "conditions": [list(c) for c in DEFAULT_CONDITIONS],
               "vis_case_id": case_ids[args.vis_case_index] if case_ids else None, "runs": summary},
              out_path.with_name("summary.json"))
    for run, conds in summary.items():
        line = " ".join(f"{c}={m['dice_mean']['mean']:.3f}" for c, m in conds.items())
        print(f"{run}: {line}", flush=True)


if __name__ == "__main__":
    main()
