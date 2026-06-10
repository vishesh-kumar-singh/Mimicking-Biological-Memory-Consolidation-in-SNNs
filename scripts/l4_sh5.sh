#!/bin/bash
EPOCHS=(1 3 5)
DATASET="NMNIST"
IS_NMNIST="--is_nmnist"
for E in "${EPOCHS[@]}"; do
    for C in 1 100 10000 1000000; do
        python scripts/run_si.py --epochs $E --runs 5 --si_c $C --si_xi 0.1 --dataset_name $DATASET $IS_NMNIST
    done
done
