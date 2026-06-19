#!/bin/bash
# run_all_split_mnist.sh - Complete Split-MNIST experiment suite
#
# Reproduces ALL experiments from the paper with Task-IL scoring enabled.
# Task A checkpoints are cached per (seed, epochs) to avoid redundant training.
#
# Experiments:
#   1. Baseline (no freezing)           -> cl_baseline.json
#   2. P-Factor Freezing (proposed)     -> freezing_{p}.json
#   3. Random Freezing (control)        -> random_{p}.json
#   4. Index Freezing (control)         -> index_{p}.json
#   5. No-Scale (ablation)              -> noscale_{p}.json
#   6. No-Reset (ablation)              -> noreset_{p}.json
#   7. Reset Zero variant (ablation)    -> reset_zero_{p}.json
#   8. Reset Scale variant (ablation)   -> reset_scale_{p}.json

set -e

RUNS=5

echo "=============================================="
echo "  Split-MNIST Full Experiment Suite"
echo "  $(date)"
echo "=============================================="

# -----------------------------------------------
# 1. Baseline (no freezing) — epochs 3
# -----------------------------------------------
echo ""
echo ">>> [1/8] BASELINE (no freezing)"
for e in 3; do
    echo "--- Baseline: epochs=$e ---"
    python -u experiments/run_baseline.py --epochs $e --runs $RUNS
done

# -----------------------------------------------
# 2. P-Factor Freezing (proposed) — epochs 3, percentiles 40-80
# -----------------------------------------------
echo ""
echo ">>> [2/8] P-FACTOR FREEZING (proposed method)"
for e in 3; do
    for p in 0.2 0.4 0.6 0.8; do
        echo "--- Freezing: epochs=$e, percentile=$p ---"
        python -u experiments/run_freezing.py --epochs $e --runs $RUNS --percentile $p
    done

# -----------------------------------------------
# 3. Random Freezing (control) — epochs 1,3,5, percentiles 0.2-0.8
# -----------------------------------------------

    for p in 0.2 0.4 0.6 0.8; do
        echo "--- Random: epochs=$e, percentile=$p ---"
        python -u experiments/run_random.py --epochs $e --runs $RUNS --percentile $p
    done

# -----------------------------------------------
# 4. PackNet Freezing (control) — epochs 3, percentiles 0.2-0.8
# -----------------------------------------------

    for p in 0.2 0.4 0.6 0.8; do
        echo "--- PackNet: epochs=$e, percentile=$p ---"
        python -u experiments/run_packnet.py --epochs $e --runs $RUNS --percentile $p
    done


# -----------------------------------------------
# 5. No-Scale ablation — epochs 5, percentiles 40,60,80
# -----------------------------------------------

    for p in 0.2 0.4 0.6 0.8; do
        echo "--- NoScale: epochs=$e, percentile=$p ---"
        python -u experiments/run_noscale.py --epochs $e --runs $RUNS --percentile $p
    done





# -----------------------------------------------
# 7. Reset Zero variant — epochs 5, percentiles 40,60,80
# -----------------------------------------------

    for p in 0.2 0.4 0.6 0.8; do
        echo "--- ResetZero: epochs=$e, percentile=$p ---"
        python -u experiments/run_reset_variants.py --epochs $e --runs $RUNS --percentile $p --reset_type zero
    done




# -----------------------------------------------
# 9. Energy & Computational Cost Analysis
# -----------------------------------------------

    echo "--- Energy comparison: epochs=$e ---"
    python -u analysis/analyze_energy.py --epochs $e --compare --seed 42
done


echo ""
echo "=============================================="
echo "  ALL SPLIT-MNIST EXPERIMENTS COMPLETE"
echo "  $(date)"
echo "=============================================="

