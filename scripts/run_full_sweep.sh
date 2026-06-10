#!/bin/bash
# Master script to run all MNIST and NMNIST experiments

echo "=================================================="
echo "   STARTING FULL EXPERIMENTAL SWEEP (MNIST + NMNIST)"
echo "=================================================="

echo ""
echo ">>> PHASE 1: SPLIT-MNIST SWEEP <<<"
bash scripts/run_all_split_mnist.sh

echo ""
echo ">>> PHASE 2: SPLIT-NMNIST SWEEP <<<"
bash scripts/run_all_nmnist.sh

echo ""
echo "=================================================="
echo "   ALL EXPERIMENTS COMPLETED SUCCESSFULLY!        "
echo "=================================================="
