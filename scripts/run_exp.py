import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import subprocess
import sys
import os
import argparse

               
EPOCHS_LIST = [1, 2, 3, 4, 5]
PERCENTILES_LIST = [0.4, 0.5, 0.6, 0.7, 0.8]
DEFAULT_RUNS = 5

def run_command(command):
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
        
                                                                                               
                                                                                      
                                                                                   
                                    
                                                                                                                  
                                                                                             
                                                                               
                                                                                                   
        
                                                                                               
        print(f"--- Running Baseline for {epochs} Epochs ---")
        cmd_baseline = f"{python_cmd} scripts/run_baseline.py --runs {args.runs} --epochs {epochs}"
        run_command(cmd_baseline)

                                             
        for p in PERCENTILES_LIST:
            print(f"--- Running Freezing (Percentile {p}) for {epochs} Epochs ---")
            cmd_freezing = f"{python_cmd} scripts/run_freezing.py --runs {args.runs} --epochs {epochs} --percentile {p}"
            run_command(cmd_freezing)
            
            print(f"--- Running Random Freezing (Percentile {p}) for {epochs} Epochs ---")
            cmd_random = f"{python_cmd} scripts/run_random.py --runs {args.runs} --epochs {epochs} --percentile {p}"
            run_command(cmd_random)
            
            print(f"--- Running Index Freezing (Percentile {p}) for {epochs} Epochs ---")
            cmd_index = f"{python_cmd} scripts/run_index.py --runs {args.runs} --epochs {epochs} --percentile {p}"
            run_command(cmd_index)
            
    print("\nAll experiments completed.")

if __name__ == "__main__":
    main()
