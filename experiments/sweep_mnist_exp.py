import os
import sys
import json
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from experiments.run_phase_b_p_factor import run_experiment as run_mnist_experiment

def main():
    ltp_values = [0.001, 0.005, 0.01]
    ltd_values = [0.001, 0.005, 0.01, 0.015]
    
    epochs = 3
    percentile = 0.8
    seeds = [42, 43, 44]  # Running on 3 seeds to average out variance!
    
    # Create exp directory
    exp_dir = Path("exp")
    exp_dir.mkdir(exist_ok=True)
    out_file = exp_dir / f"mnist_sweep_epoch{epochs}_percentile{percentile}.json"
    
    results = []
    completed_combos = set()
    if out_file.exists():
        with open(out_file, 'r') as f:
            try:
                results = json.load(f)
                for r in results:
                    completed_combos.add((r['ltp'], r['ltd']))
                print(f"Loaded {len(results)} existing results. Resuming...")
            except:
                pass
    
    print(f"Starting MNIST Hyperparameter Sweep (Epochs={epochs}, Percentile={percentile}, Seeds={seeds})...")
    for ltp in ltp_values:
        for ltd in ltd_values:
            if (ltp, ltd) in completed_combos:
                print(f"Skipping completed combination: alpha_ltp={ltp}, alpha_ltd={ltd}")
                continue
                
            print(f"\n{'='*60}")
            print(f"Testing combination: alpha_ltp={ltp}, alpha_ltd={ltd}")
            print(f"{'='*60}")
            
            combo_task_a_cil = []
            combo_task_a_til = []
            combo_task_b_cil = []
            combo_task_b_til = []
            combo_combined = []
            
            for seed in seeds:
                try:
                    print(f"  -> Running Seed {seed}...")
                    hist = run_mnist_experiment(
                        run_id=0,
                        epochs=epochs, 
                        seed=seed, 
                        percentile=percentile,
                        is_nmnist=False,  # STANDARD MNIST
                        alpha_ltp=ltp, 
                        alpha_ltd=ltd
                    )
                    
                    if hist:
                        # Extract final accuracies
                        t_a_cil = hist["full_curve"][-1] if len(hist["full_curve"]) == epochs else hist["full_curve"][2*epochs-1] if len(hist["full_curve"]) > epochs else 0.0
                        t_a_til = hist["full_curve_task_il"][-1] if "full_curve_task_il" in hist and len(hist["full_curve_task_il"]) == epochs else hist["full_curve_task_il"][2*epochs-1] if "full_curve_task_il" in hist and len(hist["full_curve_task_il"]) > epochs else 0.0
                        
                        t_b_cil = hist["task_b"][-1] if len(hist["task_b"]) > 0 else 0.0
                        t_b_til = hist["task_b_task_il"][-1] if "task_b_task_il" in hist and len(hist["task_b_task_il"]) > 0 else 0.0
                        
                        c_acc = hist.get("eval_all", 0.0)
                        
                        combo_task_a_cil.append(t_a_cil)
                        combo_task_a_til.append(t_a_til)
                        combo_task_b_cil.append(t_b_cil)
                        combo_task_b_til.append(t_b_til)
                        combo_combined.append(c_acc)
                except Exception as e:
                    print(f"Error running combination ltp={ltp}, ltd={ltd}, seed={seed}: {e}")
            
            # Average across seeds
            if combo_task_a_cil:
                res = {
                    "ltp": ltp,
                    "ltd": ltd,
                    "task_a_class_il": float(np.mean(combo_task_a_cil)),
                    "task_a_task_il": float(np.mean(combo_task_a_til)),
                    "task_b_class_il": float(np.mean(combo_task_b_cil)),
                    "task_b_task_il": float(np.mean(combo_task_b_til)),
                    "combined_acc": float(np.mean(combo_combined))
                }
                results.append(res)
                print(f"\n[Avg Result] LTP={ltp}, LTD={ltd} -> Task A (C-IL/T-IL): {res['task_a_class_il']:.2f}% / {res['task_a_task_il']:.2f}% | Task B: {res['task_b_class_il']:.2f}% / {res['task_b_task_il']:.2f}% | Combined: {res['combined_acc']:.2f}%")
                
                # Save incremental results to make it fully resumable!
                with open(out_file, "w") as f:
                    json.dump(results, f, indent=4)                
    print("\n\n=== SWEEP SUMMARY ===")
    
    # Sort results by Task A Class-IL Retention
    results.sort(key=lambda x: x['task_a_class_il'], reverse=True)
    
    for r in results:
        print(f"LTP={r['ltp']:.3f}, LTD={r['ltd']:.3f} | Task A (C-IL/T-IL): {r['task_a_class_il']:5.2f}% / {r['task_a_task_il']:5.2f}% | Task B (C-IL/T-IL): {r['task_b_class_il']:5.2f}% / {r['task_b_task_il']:5.2f}% | Combined: {r['combined_acc']:5.2f}%")
        
    # Save results to JSON
    out_file = exp_dir / f"mnist_sweep_epoch{epochs}_percentile{percentile}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nResults saved to {out_file}")

if __name__ == "__main__":
    main()
