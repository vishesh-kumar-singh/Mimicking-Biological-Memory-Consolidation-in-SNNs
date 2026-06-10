#!/bin/bash
# run_all_multitask.sh - Run 5-Task sequential learning over multiple epoch settings

echo "Starting 5-Task Fashion MNIST Sweep..."

for e in 1 2 3; do
    echo "======================================"
    echo "Running for Epochs: $e"
    echo "======================================"
    python -u scripts/run_multitask.py \
        --epochs $e \
        --runs 5 \
        --percentiles 0.2,0.3,0.4 \
        --seed_start 42 \
        --hidden_size ${HIDDEN:-4096}
done

echo "All multi-task experiments completed successfully!"
