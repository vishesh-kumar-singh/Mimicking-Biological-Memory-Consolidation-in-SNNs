#!/bin/bash
EPOCHS=(1 3 5)
for E in "${EPOCHS[@]}"; do
    for P in 0.2 0.4 0.6 0.8; do
        python -u scripts/run_freezing.py --epochs $E --runs 5 --percentile $P --data_dir ./data/nmnist --dataset_name Split-NMNIST --is_nmnist
        python -u scripts/run_random.py --epochs $E --runs 5 --percentile $P --data_dir ./data/nmnist --dataset_name Split-NMNIST --is_nmnist
    done
done
