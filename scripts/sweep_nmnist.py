import os
import sys
import argparse

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.run_nmnist import run_nmnist_experiment

def main():
    ltp_values = [0.001, 0.005, 0.01]
    ltd_values = [0.001, 0.005, 0.01, 0.015]
    
    epochs = 1
    percentile = 0.8
    seed = 42
    
    results = []
    
    print("Starting N-MNIST Hyperparameter Sweep...")
    for ltp in ltp_values:
        for ltd in ltd_values:
            print(f"\n{'='*60}")
            print(f"Testing combination: alpha_ltp={ltp}, alpha_ltd={ltd}")
            print(f"{'='*60}")
            
            # Use run_id=0 for all since we are just testing single runs per config
            hist = run_nmnist_experiment(
                run_id=0, 
                epochs=epochs, 
                seed=seed, 
                percentile=percentile, 
                alpha_ltp=ltp, 
                alpha_ltd=ltd
            )
            
            if hist:
                task_a_acc = hist["full_curve"][-1]
                task_a_retention = hist["full_curve"][-1] if len(hist["full_curve"]) == epochs else hist["full_curve"][2*epochs-1] if len(hist["full_curve"]) > epochs else 0.0
                task_b_acc = hist["task_b"][-1] if len(hist["task_b"]) > 0 else 0.0
                
                results.append({
                    "ltp": ltp,
                    "ltd": ltd,
                    "task_a_acc": task_a_acc,
                    "task_a_retention": task_a_retention,
                    "task_b_acc": task_b_acc,
                    "combined_acc": hist["eval_all"]
                })
                
                print(f"\n[Sweep Result] LTP={ltp}, LTD={ltd} -> Task A: {task_a_acc:.2f}%, Retention: {task_a_retention:.2f}%, Task B: {task_b_acc:.2f}%, Combined: {hist['eval_all']:.2f}%")
                
                # If we found a great combination, stop early!
                if task_a_acc > 80.0 and task_a_retention > 50.0 and task_b_acc > 80.0:
                    print("\n[SUCCESS] Found a working configuration!")
                    break
        else:
            continue
        break
                
    print("\n\n=== SWEEP SUMMARY ===")
    for r in results:
        status = "FAIL" if r['task_a_acc'] < 50.0 else "POOR RETENTION" if r['task_a_retention'] < 50.0 else "SUCCESS"
        print(f"LTP={r['ltp']:.3f}, LTD={r['ltd']:.3f} | Task A: {r['task_a_acc']:5.2f}% | Ret: {r['task_a_retention']:5.2f}% | Task B: {r['task_b_acc']:5.2f}% | [{status}]")

if __name__ == "__main__":
    main()
