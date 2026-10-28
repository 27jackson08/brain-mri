"""Train one configuration of the 3D U-Net on the preprocessed BraTS subset.

Example:
    python train.py --config spatial --seed 0 --epochs 40
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import torch
from monai.data import DataLoader, Dataset
from monai.losses import DiceCELoss

from common import REGION_NAMES, RUNS_DIR, SPLIT_FILE, get_device, load_json, save_json, seed_everything
from metrics import dice_per_region
from model import build_model, count_parameters, logits_to_mask, predict_logits
from transforms import CONFIGS, build_eval_transform, build_train_transform

LOG = logging.getLogger("train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", choices=sorted(CONFIGS), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--val-interval", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--samples-per-case", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-cases", type=int, default=None, help="debug: limit training cases")
    parser.add_argument("--max-val-cases", type=int, default=None, help="debug: limit validation cases")
    parser.add_argument("--run-name", default=None, help="defaults to <config>_seed<seed>")
    return parser.parse_args()


def setup_logging(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.FileHandler(run_dir / "train.log"), logging.StreamHandler()],
        force=True,
    )


def make_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    split = load_json(SPLIT_FILE)
    train_ids = split["train"][: args.max_train_cases]
    val_ids = split["val"][: args.max_val_cases]
    train_ds = Dataset(
        data=[{"case_id": c} for c in train_ids],
        transform=build_train_transform(args.config, args.samples_per_case),
    )
    val_ds = Dataset(data=[{"case_id": c} for c in val_ids], transform=build_eval_transform())
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=min(2, args.num_workers))
    LOG.info("train cases=%d (x%d patches), val cases=%d", len(train_ids), args.samples_per_case, len(val_ids))
    return train_loader, val_loader


def train_one_epoch(model, loader, loss_fn, optimizer, device) -> float:
    model.train()
    total, steps = 0.0, 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(images), labels)
        loss.backward()
        optimizer.step()
        total += loss.item()
        steps += 1
    return total / max(steps, 1)


@torch.no_grad()
def validate(model, loader, device) -> dict[str, float]:
    model.eval()
    per_region = {name: [] for name in REGION_NAMES}
    for batch in loader:
        image = batch["image"].to(device)
        label = batch["label"][0].to(device)
        pred = logits_to_mask(predict_logits(model, image))[0]
        for name, score in zip(REGION_NAMES, dice_per_region(pred, label)):
            per_region[name].append(score)
    means = {name: sum(v) / len(v) for name, v in per_region.items()}
    means["mean"] = sum(means.values()) / len(means)
    return means


def main() -> None:
    args = parse_args()
    run_name = args.run_name or f"{args.config}_seed{args.seed}"
    run_dir = RUNS_DIR / run_name
    setup_logging(run_dir)
    seed_everything(args.seed)
    device = get_device()

    train_loader, val_loader = make_loaders(args)
    model = build_model().to(device)
    # batch=True computes Dice over the whole batch, so patches where a region is
    # absent still contribute gradient instead of a constant loss of one.
    loss_fn = DiceCELoss(sigmoid=True, squared_pred=True, smooth_nr=0, smooth_dr=1e-5, batch=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    config = {**vars(args), "run_name": run_name, "device": str(device), "parameters": count_parameters(model),
              "augmentation": CONFIGS[args.config]}
    save_json(config, run_dir / "config.json")
    LOG.info("run=%s device=%s params=%.2fM config=%s", run_name, device, config["parameters"] / 1e6, args.config)

    history: list[dict] = []
    best = {"mean": -1.0, "epoch": -1}
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        scheduler.step()
        record = {"epoch": epoch, "train_loss": loss, "lr": scheduler.get_last_lr()[0],
                  "train_time_s": time.time() - epoch_start}
        if epoch % args.val_interval == 0 or epoch == args.epochs:
            val_start = time.time()
            scores = validate(model, val_loader, device)
            record.update({f"val_dice_{k}": v for k, v in scores.items()})
            record["val_time_s"] = time.time() - val_start
            if scores["mean"] > best["mean"]:
                best = {"mean": scores["mean"], "epoch": epoch, **{k: v for k, v in scores.items() if k != "mean"}}
                torch.save(model.state_dict(), run_dir / "best.pt")
            LOG.info("epoch %d/%d loss=%.4f val mean=%.4f TC=%.4f WT=%.4f ET=%.4f (best %.4f @%d) "
                     "train %.0fs val %.0fs elapsed %.1fmin", epoch, args.epochs, loss, scores["mean"], scores["TC"],
                     scores["WT"], scores["ET"], best["mean"], best["epoch"], record["train_time_s"],
                     record["val_time_s"], (time.time() - start) / 60)
            if device.type == "mps":
                torch.mps.empty_cache()
        else:
            LOG.info("epoch %d/%d loss=%.4f lr=%.2e train %.0fs elapsed %.1fmin", epoch, args.epochs, loss,
                     record["lr"], record["train_time_s"], (time.time() - start) / 60)
        history.append(record)
        save_json({"history": history, "best": best, "total_time_s": time.time() - start}, run_dir / "history.json")

    torch.save(model.state_dict(), run_dir / "last.pt")
    LOG.info("done: best val mean Dice %.4f at epoch %d, total %.1f min", best["mean"], best["epoch"],
             (time.time() - start) / 60)


if __name__ == "__main__":
    main()
