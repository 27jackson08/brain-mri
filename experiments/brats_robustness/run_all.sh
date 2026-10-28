#!/usr/bin/env bash
# End-to-end pipeline: preprocess -> train three configurations -> evaluate -> report.
# Usage: bash run_all.sh [EPOCHS] [SEED]
set -euo pipefail
cd "$(dirname "$0")"
EPOCHS="${1:-40}"
SEED="${2:-0}"

python prepare_data.py --n-train 100 --n-val 15 --n-test 30
for CONFIG in baseline spatial degradation; do
  python train.py --config "$CONFIG" --seed "$SEED" --epochs "$EPOCHS"
done
python evaluate.py --runs "baseline_seed${SEED}" "spatial_seed${SEED}" "degradation_seed${SEED}"
python report.py
