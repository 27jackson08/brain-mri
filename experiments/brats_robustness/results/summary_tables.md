# Results summary (30 held-out test cases, seed 0)

## Headline: mean Dice (average of TC, WT, ET)

| Model | Clean | Noise σ=0.2 | 2 mm resolution | Δ noise | Δ 2 mm |
|---|---|---|---|---|---|
| Baseline | 0.752 | 0.751 | 0.738 | -0.001 | -0.013 |
| Spatial Augmentation | 0.745 | 0.744 | 0.732 | -0.001 | -0.012 |
| Degradation-Aware | 0.739 | 0.738 | 0.732 | -0.000 | -0.007 |

## Clean-image accuracy per region

| Model | Dice WT | Dice TC | Dice ET | HD95 WT (vox) | HD95 TC (vox) | HD95 ET (vox) | Missed structures |
|---|---|---|---|---|---|---|---|
| Baseline | 0.797 ± 0.185 | 0.733 ± 0.186 | 0.724 ± 0.226 | 37.0 (n=30) | 28.2 (n=30) | 24.7 (n=29) | 0 |
| Spatial Augmentation | 0.784 ± 0.200 | 0.732 ± 0.188 | 0.717 ± 0.227 | 38.7 (n=30) | 26.6 (n=30) | 21.4 (n=29) | 0 |
| Degradation-Aware | 0.786 ± 0.204 | 0.725 ± 0.195 | 0.706 ± 0.237 | 40.9 (n=30) | 29.0 (n=30) | 26.3 (n=29) | 0 |

HD95 is averaged over the cases where the reference and all three models' predictions are non-empty (same n for every model). *Missed structures* counts region/case pairs where the reference is non-empty but the prediction is empty.

## Mean Dice under every degradation level (mean ± std across cases)

| Condition | Baseline | Spatial Augmentation | Degradation-Aware |
|---|---|---|---|
| Clean (1 mm) | 0.752 ± 0.164 | 0.745 ± 0.170 | 0.739 ± 0.180 |
| Gaussian noise σ=0.1 | 0.751 ± 0.164 | 0.744 ± 0.170 | 0.739 ± 0.180 |
| Gaussian noise σ=0.2 | 0.751 ± 0.165 | 0.744 ± 0.170 | 0.738 ± 0.180 |
| Gaussian noise σ=0.4 (beyond training range) | 0.751 ± 0.165 | 0.744 ± 0.171 | 0.739 ± 0.179 |
| Gaussian noise σ=0.8 (beyond training range) | 0.751 ± 0.163 | 0.744 ± 0.169 | 0.739 ± 0.176 |
| Reduced resolution ×0.5 (≈2 mm) | 0.738 ± 0.171 | 0.732 ± 0.177 | 0.732 ± 0.183 |
| Reduced resolution ×0.33 (≈3 mm, beyond training range) | 0.729 ± 0.171 | 0.725 ± 0.177 | 0.727 ± 0.180 |
| Reduced resolution ×0.25 (≈4 mm, beyond training range) | 0.716 ± 0.174 | 0.713 ± 0.179 | 0.719 ± 0.180 |

## Paired differences between models

| Condition | Spatial Augmentation − Baseline | Degradation-Aware − Spatial Augmentation | Degradation-Aware − Baseline |
|---|---|---|---|
| Clean (1 mm) | -0.007 [-0.015, +0.001] | -0.006 [-0.015, +0.002] | -0.013 [-0.027, +0.000] |
| Gaussian noise σ=0.1 | -0.007 [-0.016, +0.001] | -0.006 [-0.015, +0.002] | -0.013 [-0.027, +0.001] |
| Gaussian noise σ=0.2 | -0.007 [-0.015, +0.001] | -0.006 [-0.015, +0.002] | -0.012 [-0.027, +0.000] |
| Gaussian noise σ=0.4 (beyond training range) | -0.007 [-0.016, +0.001] | -0.005 [-0.014, +0.003] | -0.012 [-0.026, +0.000] |
| Gaussian noise σ=0.8 (beyond training range) | -0.007 [-0.017, +0.002] | -0.005 [-0.012, +0.002] | -0.012 [-0.025, -0.001]* |
| Reduced resolution ×0.5 (≈2 mm) | -0.006 [-0.013, +0.001] | -0.001 [-0.009, +0.007] | -0.006 [-0.019, +0.006] |
| Reduced resolution ×0.33 (≈3 mm, beyond training range) | -0.004 [-0.010, +0.002] | +0.002 [-0.007, +0.011] | -0.001 [-0.014, +0.010] |
| Reduced resolution ×0.25 (≈4 mm, beyond training range) | -0.002 [-0.009, +0.004] | +0.006 [-0.003, +0.015] | +0.004 [-0.010, +0.016] |

Values are mean paired differences in per-case mean Dice with 95% percentile bootstrap confidence intervals over test cases (10,000 resamples). `*` marks intervals that exclude zero. The intervals are not corrected for multiple comparisons, and all models come from a single training seed, so they describe case-to-case variability only, not training-run variability.

## Training and inference

| Model | Best val mean Dice | Best epoch | Training time | Median inference time / case |
|---|---|---|---|---|
| Baseline | 0.779 | 40 | 48 min | 2.9 s |
| Spatial Augmentation | 0.777 | 40 | 46 min | 3.1 s |
| Degradation-Aware | 0.775 | 40 | 46 min | 3.0 s |

