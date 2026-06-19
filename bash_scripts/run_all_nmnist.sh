#!/bin/bash
# run_all_nmnist.sh - Run 2-Task Split-NMNIST with the proven architecture
# Task A = classes 0-4, Task B = classes 5-9

echo "Starting 2-Task Split-NMNIST Sweep..."

# ONLY run for Epochs = 3 as requested for baselines comparison
for e in 3; do
    for p in 0.2 0.4 0.6 0.8; do
        echo "======================================"
        echo "Epochs: $e | Percentile: $p"
        echo "======================================"
        
        # Using optimal Alpha LTP/LTD from hyperparameter sweep
        echo "Running P-Factor (Ours)..."
        python -u experiments/run_nmnist.py \
            --epochs $e \
            --runs 5 \
            --percentile $p \
            --data_dir ./data/nmnist \
            --dataset_name Split-NMNIST \
            --is_nmnist \
            --alpha_ltp 0.01 \
            --alpha_ltd 0.005

        echo "Running Random Freezing Baseline..."
        python -u experiments/run_random.py \
            --epochs $e \
            --runs 5 \
            --percentile $p \
            --data_dir ./data/nmnist \
            --dataset_name Split-NMNIST \
            --is_nmnist
            
        echo "Running PackNet (Magnitude) Baseline..."
        python -u experiments/run_packnet.py \
            --epochs $e \
            --runs 5 \
            --percentile $p \
            --data_dir ./data/nmnist \
            --dataset_name Split-NMNIST \
            --is_nmnist
    done
done

echo "All Split-NMNIST experiments completed!"
