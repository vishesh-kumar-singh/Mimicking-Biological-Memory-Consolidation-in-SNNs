#!/bin/bash
# run_all_baselines.sh
# Comprehensive sweep script to run ER, EWC, and SI baselines across 
# multiple configurations, epochs, and datasets (Split-MNIST & NMNIST)
# to empirically prove they fail on SNNs compared to P-factor consolidation.

EPOCHS=(1 3 5)
EWC_LAMBDAS=(1000 100000 1000000)  # Low, Medium, High penalty
SI_CS=(1 100 10000 1000000)         # Low, Medium, High penalty
ER_BUFFERS=(200)                  # Standard 200 samples per class buffer
PACKNET_PERCENTILES=(0.2 0.4 0.6 0.8) # Fraction of network to freeze

# Run Split-MNIST first, then NMNIST
for IS_NMNIST in "" "--is_nmnist"; do
    DATASET_NAME="Split-MNIST"
    if [ "$IS_NMNIST" = "--is_nmnist" ]; then
        DATASET_NAME="NMNIST"
    fi
    
    echo "====================================================================="
    echo "STARTING SWEEP FOR DATASET: $DATASET_NAME"
    echo "====================================================================="
    
    for E in "${EPOCHS[@]}"; do
        echo "=========================================================="
        echo "Running Epochs: $E ($DATASET_NAME)"
        echo "=========================================================="
        
        # 1. Experience Replay (ER)
        for B in "${ER_BUFFERS[@]}"; do
            echo "--> Running ER (Buffer per class: $B)..."
            python scripts/run_er.py --epochs $E --runs 5 --buffer_per_class $B --dataset_name $DATASET_NAME $IS_NMNIST
        done
        
        # 2. Elastic Weight Consolidation (EWC)
        for L in "${EWC_LAMBDAS[@]}"; do
            echo "--> Running EWC (Lambda: $L)..."
            python scripts/run_ewc.py --epochs $E --runs 5 --ewc_lambda $L --dataset_name $DATASET_NAME $IS_NMNIST
        done
        
        # 3. Synaptic Intelligence (SI)
        for C in "${SI_CS[@]}"; do
            echo "--> Running SI (c parameter: $C)..."
            python scripts/run_si.py --epochs $E --runs 5 --si_c $C --si_xi 0.1 --dataset_name $DATASET_NAME $IS_NMNIST
        done
        
        # 4. PackNet (Magnitude Pruning)
        for P in "${PACKNET_PERCENTILES[@]}"; do
            echo "--> Running PackNet (Percentile: $P)..."
            python scripts/run_packnet.py --epochs $E --runs 5 --percentile $P --dataset_name $DATASET_NAME $IS_NMNIST
        done
    done
done

echo "====================================================================="
echo "ALL BASELINE RUNS COMPLETED SUCCESSFULLY"
echo "====================================================================="
