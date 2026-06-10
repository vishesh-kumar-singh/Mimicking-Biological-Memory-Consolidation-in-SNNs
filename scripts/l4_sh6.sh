#!/bin/bash
EPOCHS=(1 3 5)
for E in "${EPOCHS[@]}"; do
    python -u scripts/run_baseline.py --epochs $E --runs 5
    for P in 0.2 0.4 0.6 0.8; do
        python -u scripts/run_freezing.py --epochs $E --runs 5 --percentile $P
    done
done
