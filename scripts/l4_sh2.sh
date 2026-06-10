#!/bin/bash
EPOCHS=(1 3 5)
DATASET="Split-MNIST"
for E in "${EPOCHS[@]}"; do
    for L in 1000 100000 1000000; do
        python scripts/run_ewc.py --epochs $E --runs 5 --ewc_lambda $L --dataset_name $DATASET
    done
done
