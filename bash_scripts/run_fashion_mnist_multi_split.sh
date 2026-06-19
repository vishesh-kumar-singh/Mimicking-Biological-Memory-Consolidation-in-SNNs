#!/bin/bash
# run_fashion_mnist_multi_split.sh
# Run 5-split sweeps for P-Factor, ER Baseline, and Fine-Tuning Baseline.

echo "======================================================"
echo "Starting 5-Split Fashion-MNIST Sweeps"
echo "======================================================"

# --- 5-Split Sweep ---
echo ""
echo ">>> Starting 5-Split Sweep (P-Factor) <<<"
python -u experiments/run_fashion_mnist_sweep.py \
    --mode single \
    --percentile 0.15 \
    --alpha_ltp 0.01 \
    --alpha_ltd 0.005 \
    --num_tasks 5 \
    --runs 5 \
    --epochs 3 \
    --seed_start 42

echo ""
echo ">>> Starting 5-Split Sweep (ER Baseline) <<<"
python -u experiments/run_fashion_mnist_sweep.py \
    --mode single \
    --er_mode \
    --er_buffer 200 \
    --num_tasks 5 \
    --runs 5 \
    --epochs 3 \
    --seed_start 42

echo ""
echo ">>> Starting 5-Split Sweep (Fine-Tuning Baseline) <<<"
python -u experiments/run_fashion_mnist_sweep.py \
    --mode single \
    --baseline_mode \
    --num_tasks 5 \
    --runs 5 \
    --epochs 3 \
    --seed_start 42

echo ""
echo ">>> Starting 5-Split Sweep (Random Freezing) <<<"
python -u experiments/run_fashion_mnist_sweep.py \
    --mode single \
    --random_mode \
    --percentile 0.15 \
    --num_tasks 5 \
    --runs 5 \
    --epochs 3 \
    --seed_start 42

echo ""
echo ">>> Starting 5-Split Sweep (PackNet) <<<"
python -u experiments/run_fashion_mnist_sweep.py \
    --mode single \
    --packnet_mode \
    --percentile 0.15 \
    --num_tasks 5 \
    --runs 5 \
    --epochs 3 \
    --seed_start 42

echo ""
echo ">>> Starting 5-Split Sweep (EWC) <<<"
for lambda_val in 1000 100000 1000000; do
    echo "Running EWC (lambda=${lambda_val})..."
    python -u experiments/run_fashion_mnist_sweep.py \
        --mode single \
        --ewc_mode \
        --ewc_lambda ${lambda_val} \
        --num_tasks 5 \
        --runs 5 \
        --epochs 3 \
        --seed_start 42
done

echo ""
echo ">>> Starting 5-Split Sweep (SI) <<<"
for c_val in 1 100 10000 1000000; do
    echo "Running SI (c=${c_val})..."
    python -u experiments/run_fashion_mnist_sweep.py \
        --mode single \
        --si_mode \
        --si_c ${c_val} \
        --num_tasks 5 \
        --runs 5 \
        --epochs 3 \
        --seed_start 42
done

echo ""
echo "======================================================"
echo "All 5-split sweeps completed successfully!"
echo "======================================================"
