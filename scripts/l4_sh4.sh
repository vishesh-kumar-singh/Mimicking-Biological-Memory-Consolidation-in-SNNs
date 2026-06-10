#!/bin/bash
EPOCHS=(1 3 5)
DATASET="NMNIST"
IS_NMNIST="--is_nmnist"
for E in "${EPOCHS[@]}"; do
    python scripts/run_er.py --epochs $E --runs 5 --buffer_per_class 200 --dataset_name $DATASET $IS_NMNIST
    for P in 0.2 0.4 0.6 0.8; do
        python scripts/run_packnet.py --epochs $E --runs 5 --percentile $P --dataset_name $DATASET $IS_NMNIST
    done
    for L in 1000 100000 1000000; do
        python scripts/run_ewc.py --epochs $E --runs 5 --ewc_lambda $L --dataset_name $DATASET $IS_NMNIST
    done
done
