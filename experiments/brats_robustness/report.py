"""Turn evaluation outputs into markdown tables and figures for the README.

Reads ``results/per_case_results.csv`` plus each run's ``history.json`` and
writes ``results/summary_tables.md``, ``results/paired_differences.json`` and
three figures under ``results/figures``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import REGION_NAMES, RESULTS_DIR, RUNS_DIR, SPLIT_FILE, load_json, save_json  # noqa: E402
from degradations import apply_degradation  # noqa: E402
from evaluate import NOISE_SEED_BASE  # noqa: E402
from transforms import CONFIGS, LoadPreprocessedCased  # noqa: E402

CONFIG_ORDER = ("baseline", "spatial", "degradation")
MODEL_COLORS = {"baseline": "#2a78d6", "spatial": "#eb6834", "degradation": "#1baf7a"}
MODEL_MARKERS = {"baseline": "o", "spatial": "s", "degradation": "^"}
REGION_COLORS = {"edema": "#6250d6", "core": "#eda100", "enhancing": "#e34948"}
HEADLINE_CONDITIONS = (("clean", "Clean"), ("noise_0.2", "Noise σ=0.2"), ("lowres_0.5", "2 mm resolution"))
CONDITION_LABELS = {
    "clean": "Clean (1 mm)", "noise_0.1": "Gaussian noise σ=0.1", "noise_0.2": "Gaussian noise σ=0.2",
    "noise_0.4": "Gaussian noise σ=0.4 (beyond training range)", "lowres_0.5": "Reduced resolution ×0.5 (≈2 mm)",
    "lowres_0.333": "Reduced resolution ×0.33 (≈3 mm, beyond training range)",
}
BOOTSTRAP_SAMPLES = 10_000
TEXT_COLOR, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def style_axis(ax: plt.Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(TEXT_COLOR)


def label_of(config: str) -> str:
    return CONFIGS[config]["label"]


def load_histories(seed: int) -> dict[str, dict]:
    out = {}
    for config in CONFIG_ORDER:
        path = RUNS_DIR / f"{config}_seed{seed}" / "history.json"
        if path.exists():
            out[config] = load_json(path)
    return out


def paired_bootstrap(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> dict[str, float]:
    """Mean of (a - b) with a percentile bootstrap CI over paired cases."""
    diff = a - b
    idx = rng.integers(0, len(diff), size=(BOOTSTRAP_SAMPLES, len(diff)))
    means = diff[idx].mean(axis=1)
    return {"mean": float(diff.mean()), "ci_low": float(np.percentile(means, 2.5)),
            "ci_high": float(np.percentile(means, 97.5)), "n": int(len(diff))}


def case_matrix(df: pd.DataFrame, condition: str, metric: str) -> pd.DataFrame:
    """Cases x configs table for one condition; requires exactly one row per (case, config)."""
    sub = df[df.condition == condition]
    matrix = sub.pivot(index="case_id", columns="config", values=metric).sort_index()
    missing = [c for c in CONFIG_ORDER if c not in matrix.columns]
    if missing or matrix.isna().any().any():
        raise ValueError(f"incomplete results for condition {condition}: missing configs {missing} or cases")
    return matrix


def fmt(mean: float, std: float | None = None) -> str:
    return f"{mean:.3f}" if std is None else f"{mean:.3f} ± {std:.3f}"


def headline_table(df: pd.DataFrame) -> list[str]:
    lines = ["| Model | " + " | ".join(lbl for _, lbl in HEADLINE_CONDITIONS) + " | Δ noise | Δ 2 mm |",
             "|---|" + "---|" * (len(HEADLINE_CONDITIONS) + 2)]
    for config in CONFIG_ORDER:
        means = {c: df[(df.config == config) & (df.condition == c)].dice_mean.mean() for c, _ in HEADLINE_CONDITIONS}
        cells = [fmt(means[c]) for c, _ in HEADLINE_CONDITIONS]
        cells += [f"{means['noise_0.2'] - means['clean']:+.3f}", f"{means['lowres_0.5'] - means['clean']:+.3f}"]
        lines.append(f"| {label_of(config)} | " + " | ".join(cells) + " |")
    return lines


def region_table(df: pd.DataFrame, condition: str = "clean") -> list[str]:
    """Per-region Dice and HD95; HD95 is averaged over the cases valid for every model."""
    sub_all = df[df.condition == condition]
    valid_cases = {r: set(sub_all.case_id) for r in REGION_NAMES}
    for r in REGION_NAMES:
        for config in CONFIG_ORDER:
            ok = sub_all[(sub_all.config == config) & sub_all[f"hd95_{r}"].notna()].case_id
            valid_cases[r] &= set(ok)
    lines = ["| Model | Dice WT | Dice TC | Dice ET | HD95 WT (vox) | HD95 TC (vox) | HD95 ET (vox) | Missed structures |",
             "|---|---|---|---|---|---|---|---|"]
    for config in CONFIG_ORDER:
        sub = sub_all[sub_all.config == config]
        cells = [fmt(sub[f"dice_{r}"].mean(), sub[f"dice_{r}"].std(ddof=0)) for r in ("WT", "TC", "ET")]
        for r in ("WT", "TC", "ET"):
            vals = sub[sub.case_id.isin(valid_cases[r])][f"hd95_{r}"]
            cells.append(f"{vals.mean():.1f} (n={len(vals)})" if len(vals) else "n/a")
        missed = sum(int(((sub[f"gt_voxels_{r}"] > 0) & (sub[f"pred_voxels_{r}"] == 0)).sum()) for r in REGION_NAMES)
        cells.append(str(missed))
        lines.append(f"| {label_of(config)} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("HD95 is averaged over the cases where the reference and all three models' predictions are "
                 "non-empty (same n for every model). *Missed structures* counts region/case pairs where the "
                 "reference is non-empty but the prediction is empty.")
    return lines


def full_grid_table(df: pd.DataFrame) -> list[str]:
    conditions = [c for c in CONDITION_LABELS if c in set(df.condition)]
    lines = ["| Condition | " + " | ".join(label_of(c) for c in CONFIG_ORDER) + " |", "|---|" + "---|" * len(CONFIG_ORDER)]
    for cond in conditions:
        cells = []
        for config in CONFIG_ORDER:
            sub = df[(df.config == config) & (df.condition == cond)].dice_mean
            cells.append(fmt(sub.mean(), sub.std(ddof=0)))
        lines.append(f"| {CONDITION_LABELS[cond]} | " + " | ".join(cells) + " |")
    return lines


def paired_table(df: pd.DataFrame, rng: np.random.Generator) -> tuple[list[str], dict]:
    pairs = (("spatial", "baseline"), ("degradation", "spatial"), ("degradation", "baseline"))
    conditions = [c for c in CONDITION_LABELS if c in set(df.condition)]
    results: dict = {}
    lines = ["| Condition | " + " | ".join(f"{label_of(a)} − {label_of(b)}" for a, b in pairs) + " |",
             "|---|" + "---|" * len(pairs)]
    for cond in conditions:
        matrix = case_matrix(df, cond, "dice_mean")
        cells = []
        for a, b in pairs:
            stats = paired_bootstrap(matrix[a].to_numpy(), matrix[b].to_numpy(), rng)
            results.setdefault(cond, {})[f"{a}-{b}"] = stats
            marker = "*" if stats["ci_low"] > 0 or stats["ci_high"] < 0 else ""
            cells.append(f"{stats['mean']:+.3f} [{stats['ci_low']:+.3f}, {stats['ci_high']:+.3f}]{marker}")
        lines.append(f"| {CONDITION_LABELS[cond]} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Values are mean paired differences in per-case mean Dice with 95% percentile bootstrap "
                 f"confidence intervals over test cases ({BOOTSTRAP_SAMPLES:,} resamples). "
                 "`*` marks intervals that exclude zero. The intervals are not corrected for multiple "
                 "comparisons, and all models come from a single training seed, so they describe "
                 "case-to-case variability only, not training-run variability.")
    return lines, results


def training_table(histories: dict[str, dict], df: pd.DataFrame) -> list[str]:
    lines = ["| Model | Best val mean Dice | Best epoch | Training time | Median inference time / case |",
             "|---|---|---|---|---|"]
    for config in CONFIG_ORDER:
        h = histories.get(config)
        infer = df[(df.config == config) & (df.condition == "clean")].infer_time_s.median()
        if h is None:
            lines.append(f"| {label_of(config)} | n/a | n/a | n/a | {infer:.1f} s |")
            continue
        lines.append(f"| {label_of(config)} | {h['best']['mean']:.3f} | {h['best']['epoch']} | "
                     f"{h['total_time_s'] / 60:.0f} min | {infer:.1f} s |")
    return lines


def plot_training_curves(histories: dict[str, dict], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), dpi=150)
    for config, h in histories.items():
        hist = pd.DataFrame(h["history"])
        axes[0].plot(hist.epoch, hist.train_loss, color=MODEL_COLORS[config], linewidth=2, label=label_of(config))
        if "val_dice_mean" not in hist.columns:
            continue
        val = hist.dropna(subset=["val_dice_mean"])
        axes[1].plot(val.epoch, val["val_dice_mean"], color=MODEL_COLORS[config], linewidth=2,
                     marker=MODEL_MARKERS[config], markersize=5, label=label_of(config))
    axes[0].set(title="Training loss (Dice + BCE)", xlabel="Epoch", ylabel="Loss")
    axes[1].set(title="Validation mean Dice (TC, WT, ET)", xlabel="Epoch", ylabel="Dice")
    for ax in axes:
        style_axis(ax)
    axes[1].legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    plt.close(fig)


def plot_robustness(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), dpi=150)
    panels = (
        (axes[0], [("clean", 0.0), ("noise_0.1", 0.1), ("noise_0.2", 0.2), ("noise_0.4", 0.4)],
         "Gaussian noise σ (normalized units)", "Dice under additive noise"),
        (axes[1], [("clean", 1.0), ("lowres_0.5", 2.0), ("lowres_0.333", 3.0)],
         "Simulated voxel size (mm)", "Dice under reduced resolution"),
    )
    for ax, points, xlabel, title in panels:
        for config in CONFIG_ORDER:
            xs, ys, errs = [], [], []
            for cond, x in points:
                vals = df[(df.config == config) & (df.condition == cond)].dice_mean
                if vals.empty:
                    continue
                xs.append(x)
                ys.append(vals.mean())
                errs.append(1.96 * vals.std(ddof=1) / np.sqrt(len(vals)))
            ax.errorbar(xs, ys, yerr=errs, color=MODEL_COLORS[config], linewidth=2, marker=MODEL_MARKERS[config],
                        markersize=6, capsize=3, label=label_of(config))
        ax.set(title=title, xlabel=xlabel, ylabel="Mean Dice (test set)")
        ax.set_xticks([x for _, x in points])
        style_axis(ax)
    axes[0].legend(frameon=False, fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    plt.close(fig)


def overlay(ax: plt.Axes, base: np.ndarray, regions: np.ndarray | None, title: str) -> None:
    ax.imshow(np.rot90(base), cmap="gray", vmin=-2.5, vmax=2.5)
    if regions is not None:
        tc, wt, et = regions.astype(bool)
        for mask, color in ((wt & ~tc, REGION_COLORS["edema"]), (tc & ~et, REGION_COLORS["core"]), (et, REGION_COLORS["enhancing"])):
            rgba = np.zeros((*mask.shape, 4))
            rgba[mask] = matplotlib.colors.to_rgba(color, alpha=0.55)
            ax.imshow(np.rot90(rgba))
    ax.set_title(title, fontsize=8, color=TEXT_COLOR)
    ax.axis("off")


def plot_qualitative(df: pd.DataFrame, case_id: str, seed: int, split_name: str, out: Path) -> None:
    split = load_json(SPLIT_FILE)
    case_index = split[split_name].index(case_id)
    data = LoadPreprocessedCased()({"case_id": case_id})
    image, label = data["image"], data["label"].numpy().astype(bool)
    z = int(np.argmax(label[1].sum(axis=(0, 1))))
    rows = (("clean", "clean", 1.0), ("noise_0.2", "noise", 0.2), ("lowres_0.5", "lowres", 0.5))
    fig, axes = plt.subplots(len(rows), 3 + len(CONFIG_ORDER), figsize=(13, 2.6 * len(rows) + 0.4), dpi=110)
    for r, (cond, kind, level) in enumerate(rows):
        degraded = apply_degradation(image, kind, level, seed=NOISE_SEED_BASE + case_index).numpy()
        overlay(axes[r, 0], degraded[0, :, :, z], None, f"FLAIR\n{CONDITION_LABELS[cond]}")
        overlay(axes[r, 1], degraded[2, :, :, z], None, "T1gd")
        overlay(axes[r, 2], degraded[0, :, :, z], label[:, :, :, z], "Reference")
        for c, config in enumerate(CONFIG_ORDER):
            path = RUNS_DIR / "vis" / f"{config}_seed{seed}__{case_id}__{cond}.npy"
            pred = np.load(path)[:, :, :, z] if path.exists() else None
            dice = df[(df.config == config) & (df.condition == cond) & (df.case_id == case_id)].dice_mean
            score = f" (Dice {dice.iloc[0]:.2f})" if len(dice) else ""
            overlay(axes[r, 3 + c], degraded[0, :, :, z], pred, f"{label_of(config)}{score}")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.75) for c in REGION_COLORS.values()]
    fig.legend(handles, ["Edema (WT only)", "Non-enhancing core (TC)", "Enhancing tumour (ET)"], loc="lower center",
               ncol=3, frameon=False, fontsize=9)
    fig.suptitle(f"Held-out case {case_id}, axial slice {z}", fontsize=10, color=TEXT_COLOR)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97), h_pad=2.0)
    fig.savefig(out, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results", default=str(RESULTS_DIR / "per_case_results.csv"))
    args = parser.parse_args()
    df = pd.read_csv(args.results)
    df = df[df.seed == args.seed]
    if df.empty:
        raise SystemExit(f"no rows with seed {args.seed} in {args.results}")
    summary = load_json(Path(args.results).with_name("summary.json"))
    histories = load_histories(args.seed)
    rng = np.random.default_rng(0)
    out_dir = Path(args.results).parent
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    paired_lines, paired = paired_table(df, rng)
    n_cases = df.case_id.nunique()
    sections = [
        f"# Results summary ({n_cases} held-out test cases, seed {args.seed})", "",
        "## Headline: mean Dice (average of TC, WT, ET)", "", *headline_table(df), "",
        "## Clean-image accuracy per region", "", *region_table(df), "",
        "## Mean Dice under every degradation level (mean ± std across cases)", "", *full_grid_table(df), "",
        "## Paired differences between models", "", *paired_lines, "",
        "## Training and inference", "", *training_table(histories, df), "",
    ]
    (out_dir / "summary_tables.md").write_text("\n".join(sections) + "\n")
    save_json(paired, out_dir / "paired_differences.json")

    if histories:
        plot_training_curves(histories, figures / "training_curves.png")
    plot_robustness(df, figures / "robustness.png")
    if summary.get("vis_case_id"):
        plot_qualitative(df, summary["vis_case_id"], args.seed, summary.get("split", "test"),
                         figures / "qualitative.png")
    print("\n".join(sections))


if __name__ == "__main__":
    main()
