#!/bin/bash
EPOCHS=(1 3 5)
for E in "${EPOCHS[@]}"; do
    for P in 0.2 0.4 0.6 0.8; do
        python -u scripts/run_reset_variants.py --epochs $E --runs 5 --percentile $P --reset_type zero
        python -u scripts/run_reset_variants.py --epochs $E --runs 5 --percentile $P --reset_type scale
    done
done
