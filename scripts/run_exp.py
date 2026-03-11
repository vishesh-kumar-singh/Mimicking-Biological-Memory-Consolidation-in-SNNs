"""
run_exp.py - Master Experiment Runner

Orchestrates all SNN continual learning experiments. Runs each experiment
type across all epoch/percentile configurations. Each sub-script is
idempotent (skips already-completed runs), so this can be safely re-run.

Experiment Types:
-----------------
1. Baseline:   No freezing (catastrophic forgetting control)
2. Freezing:   P-factor based engram freezing (proposed method)
3. Random:     Random neuron freezing (control - tests if WHICH neurons matter)
4. Index:      Index-based neuron freezing (control - tests deterministic selection)
5. NoScale:    P-factor tracking without weight scaling (ablation - isolates
               P-factor identification vs weight modulation)

Usage:
------
    python scripts/run_exp.py --runs 5
    python scripts/run_exp.py --runs 5 --python_path /path/to/python
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import subprocess
import argparse

# =============================================================================
# Experiment Configuration
# =============================================================================
EPOCHS_LIST = [1, 2, 3, 4, 5]
PERCENTILES_LIST = [0.4, 0.5, 0.6, 0.7, 0.8]
DEFAULT_RUNS = 5


def run_command(command):
    """Execute a shell command, printing errors but continuing on failure."""
    print(f"Running: {command}")
    try:
        subprocess.run(command, shell=True, check=True, executable="/bin/bash")
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}")
        print(e)


def main():
    parser = argparse.ArgumentParser(description="Automate SNN Experiments")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS, help="Number of runs per configuration")
    parser.add_argument("--python_path", type=str, default=sys.executable, help="Path to python executable")
    args = parser.parse_args()

    python_cmd = args.python_path

    print(f"Starting Experiments with {args.runs} runs each...")
    print(f"Epochs: {EPOCHS_LIST}")
    print(f"Percentiles: {PERCENTILES_LIST}")
    print("="*50)

    for epochs in EPOCHS_LIST:
        print(f"\n>>> Starting Experiments for {epochs} Epochs <<<")
        
        # Step 1: Baseline (no freezing) - shows catastrophic forgetting
        print(f"--- Running Baseline for {epochs} Epochs ---")
        cmd_baseline = f"{python_cmd} scripts/run_baseline.py --runs {args.runs} --epochs {epochs}"
        run_command(cmd_baseline)

        # Step 2: For each percentile, run all freezing variants
        for p in PERCENTILES_LIST:
            # P-factor freezing (proposed method)
            print(f"--- Running Freezing (Percentile {p}) for {epochs} Epochs ---")
            cmd_freezing = f"{python_cmd} scripts/run_freezing.py --runs {args.runs} --epochs {epochs} --percentile {p}"
            run_command(cmd_freezing)
            
            # Random freezing (control: tests if selection method matters)
            print(f"--- Running Random Freezing (Percentile {p}) for {epochs} Epochs ---")
            cmd_random = f"{python_cmd} scripts/run_random.py --runs {args.runs} --epochs {epochs} --percentile {p}"
            run_command(cmd_random)
            
            # Index freezing (control: deterministic but arbitrary selection)
            print(f"--- Running Index Freezing (Percentile {p}) for {epochs} Epochs ---")
            cmd_index = f"{python_cmd} scripts/run_index.py --runs {args.runs} --epochs {epochs} --percentile {p}"
            run_command(cmd_index)
            
            # NoScale freezing (ablation: P-factor identification without weight modulation)
            print(f"--- Running NoScale Freezing (Percentile {p}) for {epochs} Epochs ---")
            cmd_noscale = f"{python_cmd} scripts/run_noscale.py --runs {args.runs} --epochs {epochs} --percentile {p}"
            run_command(cmd_noscale)
            
    print("\nAll experiments completed.")

if __name__ == "__main__":
    main()
