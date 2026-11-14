#!/usr/bin/env bash
# Reproduce every table and figure of the paper (about 45 minutes on 2 CPU cores, no GPU).
set -euo pipefail
python experiments/01_dataset_stats.py
python experiments/02_benchmark.py
python experiments/03_ablation.py LightGBM
python experiments/04_anomaly.py
python experiments/05_pipeline.py
python experiments/06_cost.py
python experiments/07_figures.py
