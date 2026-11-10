#!/usr/bin/env bash
# Full pipeline. Usage: GPU_HOURLY_USD=0.35 bash run_all.sh
set -euo pipefail
cd "$(dirname "$0")"
python src/prepare_data.py
python src/train.py --gpu_hourly_usd "${GPU_HOURLY_USD:-0}" 2>&1 | tee logs/train_stdout.log
python src/evaluate.py 2>&1 | tee logs/eval_stdout.log
python src/report.py
