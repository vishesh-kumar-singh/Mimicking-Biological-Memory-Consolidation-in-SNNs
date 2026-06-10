#!/bin/bash
EPOCHS=(1 3 5)
DATASET="Split-MNIST"
for E in "${EPOCHS[@]}"; do
    python scripts/run_er.py --epochs $E --runs 5 --buffer_per_class 200 --dataset_name $DATASET
    for P in 0.2 0.4 0.6 0.8; do
        python scripts/run_packnet.py --epochs $E --runs 5 --percentile $P --dataset_name $DATASET
    done
done
