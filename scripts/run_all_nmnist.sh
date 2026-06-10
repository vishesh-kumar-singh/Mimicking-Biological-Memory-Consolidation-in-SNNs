#!/bin/bash
# run_all_nmnist.sh - Run 2-Task Split-NMNIST with the proven architecture
# Task A = classes 0-4, Task B = classes 5-9

echo "Starting 2-Task Split-NMNIST Sweep..."

for e in 1 3 5; do
    for p in 0.2 0.4 0.6 0.8; do
        echo "======================================"
        echo "Epochs: $e | Percentile: $p"
        echo "======================================"
        python -u scripts/run_freezing.py \
            --epochs $e \
            --runs 5 \
            --percentile $p \
            --data_dir ./data/nmnist \
            --dataset_name Split-NMNIST \
            --is_nmnist

        echo "Running Random Freezing Baseline..."
        python -u scripts/run_random.py \
            --epochs $e \
            --runs 5 \
            --percentile $p \
            --data_dir ./data/nmnist \
            --dataset_name Split-NMNIST \
            --is_nmnist
    done
done

echo "All Split-NMNIST experiments completed!"
