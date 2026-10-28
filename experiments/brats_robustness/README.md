# BraTS robustness experiment

Does data augmentation make a 3D U-Net more robust to degraded MRI quality,
and at what cost to clean-image accuracy? This directory holds the full,
runnable pipeline behind the results reported in the top-level README.

Everything here was run on a single Apple M3 laptop (16 GB, PyTorch MPS
backend), so the experiment is deliberately scaled down relative to a
full BraTS training run. The protocol, splits, seeds, and degradations are
fixed in code so that the numbers can be reproduced or scaled up.

## Protocol

| Item | Setting |
|---|---|
| Dataset | Medical Segmentation Decathlon Task01 BrainTumour (BraTS 2016/17), 484 labelled cases, 4 modalities (FLAIR, T1w, T1gd, T2w), 1 mm isotropic |
| Split | 100 train / 15 validation / 30 held-out test, drawn once with seed 0 (`data/preprocessed/split.json`) |
| Targets | Nested BraTS regions: whole tumour (WT), tumour core (TC), enhancing tumour (ET) |
| Preprocessing | RAS reorientation, crop to brain bounding box, per-modality z-score over non-zero voxels |
| Model | MONAI 3D U-Net, channels (16, 32, 64, 128, 256), strides 2, two residual units per level, instance norm, 4.8 M parameters |
| Patches | 64³ training patches, 2 per case per epoch: an 80³ foreground-biased crop (1:1 tumour-centred vs random) is augmented, then centre-cropped to 64³ |
| Loss / optimiser | Dice + BCE (sigmoid, multi-label), Adam lr 5e-4, weight decay 1e-5, cosine schedule |
| Budget | 40 epochs × 200 patches, batch 2, identical for every configuration; validation every 4 epochs, best-validation checkpoint kept |
| Inference | Sliding window 64³, 50% overlap, Gaussian blending, sigmoid > 0.5 |

### Configurations

The three configurations form an additive ladder, so each adjacent pair
isolates one factor.

| Name | Training augmentation |
|---|---|
| Baseline | random crops only |
| Spatial Augmentation | + random affine (±15° rotation, ±10% scale, p=0.5) and random flips on all three axes |
| Degradation-Aware | + Gaussian noise (σ ~ U[0, 0.3] of the normalized intensity scale, p=0.5) and simulated low resolution (isotropic downsample by a factor in [0.5, 1] with area interpolation, trilinear upsample back, p=0.5) |

### Test-time degradations

Degradations are applied to the normalized test volume with a fixed seed per
case, so every model is scored on exactly the same corrupted inputs (paired
design). Reference labels stay at 1 mm.

| Condition | Levels evaluated | Inside the training range of the degradation-aware model? |
|---|---|---|
| Gaussian noise | σ = 0.1, 0.2, 0.4 | 0.1 and 0.2 yes; 0.4 no |
| Reduced resolution | factor 0.5 (≈2 mm voxels), 0.333 (≈3 mm voxels) | 0.5 at the edge; 0.333 no |

Metrics per case and region: Dice (BraTS convention, empty-vs-empty = 1) and
95th-percentile Hausdorff distance in voxels (undefined when either mask is
empty; those cases are excluded and counted).

## Files

| File | Role |
|---|---|
| `prepare_data.py` | extract the MSD tarball, build the split, preprocess cases to `.npy` |
| `transforms.py` | the three augmentation pipelines |
| `train.py` | train one configuration and seed |
| `evaluate.py` | paired robustness evaluation on the test split |
| `report.py` | tables, bootstrap confidence intervals, and figures |
| `degradations.py`, `metrics.py`, `model.py`, `common.py` | shared pieces |
| `tests/` | unit tests for label mapping, metrics, degradations, and transform shapes |
| `results/` | committed outputs: per-case CSV, summary JSON, markdown tables, figures |
| `run_all.sh` | the whole pipeline in order |

## Reproducing

```bash
# from the repository root
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

mkdir -p data/raw
curl -o data/raw/Task01_BrainTumour.tar \
  https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar

cd experiments/brats_robustness
python -m pytest tests -q
bash run_all.sh 40 0
```

Data, checkpoints, and the virtual environment are git-ignored. Set
`BRATS_DATA_ROOT` to keep the data elsewhere.
