import os
import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from experiments.run_nmnist import run_experiment as run_nmnist_experiment

def main():
    ltp_values = [0.001, 0.005, 0.01]
    ltd_values = [0.001, 0.005, 0.01, 0.015]
    
    epochs = 3
    percentile = 0.8
    seed = 42
    
    # Create exp directory
    exp_dir = Path("exp")
    exp_dir.mkdir(exist_ok=True)
    
    results = []
    
    print(f"Starting N-MNIST Hyperparameter Sweep (Epochs={epochs}, Percentile={percentile})...")
    for ltp in ltp_values:
        for ltd in ltd_values:
            print(f"\n{'='*60}")
            print(f"Testing combination: alpha_ltp={ltp}, alpha_ltd={ltd}")
            print(f"{'='*60}")
            
            # Use run_id corresponding to the ltp/ltd combination
            run_id = f"ltp{ltp}_ltd{ltd}"
            
            try:
                hist = run_nmnist_experiment(
                    run_id=0, # Just run it, the checkpoints handle it via seed+ltp+ltd
                    epochs=epochs, 
                    seed=seed, 
                    percentile=percentile,
                    is_nmnist=True,
                    alpha_ltp=ltp, 
                    alpha_ltd=ltd
                )
                
                if hist:
                    task_a_acc_class_il = hist["full_curve"][-1] if len(hist["full_curve"]) == epochs else hist["full_curve"][2*epochs-1] if len(hist["full_curve"]) > epochs else 0.0
                    task_a_acc_task_il = hist["full_curve_task_il"][-1] if "full_curve_task_il" in hist and len(hist["full_curve_task_il"]) == epochs else hist["full_curve_task_il"][2*epochs-1] if "full_curve_task_il" in hist and len(hist["full_curve_task_il"]) > epochs else 0.0
                    
                    task_b_acc_class_il = hist["task_b"][-1] if len(hist["task_b"]) > 0 else 0.0
                    task_b_acc_task_il = hist["task_b_task_il"][-1] if "task_b_task_il" in hist and len(hist["task_b_task_il"]) > 0 else 0.0
                    
                    combined_acc = hist.get("eval_all", 0.0)
                    
                    res = {
                        "ltp": ltp,
                        "ltd": ltd,
                        "task_a_class_il": task_a_acc_class_il,
                        "task_a_task_il": task_a_acc_task_il,
                        "task_b_class_il": task_b_acc_class_il,
                        "task_b_task_il": task_b_acc_task_il,
                        "combined_acc": combined_acc
                    }
                    results.append(res)
                    
                    print(f"\n[Sweep Result] LTP={ltp}, LTD={ltd} -> Task A (Class-IL/Task-IL): {task_a_acc_class_il:.2f}% / {task_a_acc_task_il:.2f}% | Task B: {task_b_acc_class_il:.2f}% / {task_b_acc_task_il:.2f}% | Combined: {combined_acc:.2f}%")
            except Exception as e:
                print(f"Error running combination ltp={ltp}, ltd={ltd}: {e}")
                
    print("\n\n=== SWEEP SUMMARY ===")
    
    # Sort results by Task A Task-IL Retention
    results.sort(key=lambda x: x['task_a_task_il'], reverse=True)
    
    for r in results:
        print(f"LTP={r['ltp']:.3f}, LTD={r['ltd']:.3f} | Task A (C-IL/T-IL): {r['task_a_class_il']:5.2f}% / {r['task_a_task_il']:5.2f}% | Task B (C-IL/T-IL): {r['task_b_class_il']:5.2f}% / {r['task_b_task_il']:5.2f}% | Combined: {r['combined_acc']:5.2f}%")
        
    # Save results to JSON
    out_file = exp_dir / f"nmnist_sweep_epoch{epochs}_percentile{percentile}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nResults saved to {out_file}")

if __name__ == "__main__":
    main()
