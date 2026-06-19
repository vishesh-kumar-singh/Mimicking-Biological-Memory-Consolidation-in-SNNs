#!/bin/bash
# run_all_nmnist_baselines.sh

echo "Running Fine-Tuning (CL Baseline)..."
python -u experiments/run_baseline.py --epochs 3 --runs 5 --data_dir ./data/nmnist --dataset_name Split-NMNIST --is_nmnist

echo "Running Experience Replay (ER)..."
python -u experiments/run_er.py --epochs 3 --runs 5 --data_dir ./data/nmnist --dataset_name Split-NMNIST --is_nmnist --buffer_per_class 200

echo "Running EWC Sweeps..."
for lambda_val in 1000 100000 1000000; do
    echo "Running EWC (lambda=${lambda_val})..."
    python -u experiments/run_ewc.py --epochs 3 --runs 5 --data_dir ./data/nmnist --dataset_name Split-NMNIST --is_nmnist --ewc_lambda $lambda_val
done

echo "Running SI Sweeps..."
for c_val in 1 100 10000 1000000; do
    echo "Running SI (c=${c_val})..."
    python -u experiments/run_si.py --epochs 3 --runs 5 --data_dir ./data/nmnist --dataset_name Split-NMNIST --is_nmnist --si_c $c_val
done

echo "Done running N-MNIST baselines."
