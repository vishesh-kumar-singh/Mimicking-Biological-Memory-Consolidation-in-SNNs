import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
import json
import os
import numpy as np

def load_results(filename):
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found.")
        return None
    with open(filename, "r") as f:
        return json.load(f)

def plot_results():
               
    baseline_data = load_results("results/cl_baseline.json")
    freezing_data = load_results("results/freezing_70.json")
    
    if not baseline_data or not freezing_data:
        print("Could not load results. Make sure 'results/cl_baseline.json' and 'results/freezing_70.json' exist.")
        return

                                                        
    plt.figure(figsize=(10, 6), dpi=150)
    
                              
    if "average" in baseline_data and "full_curve_mean" in baseline_data["average"]:
        base_mean = np.array(baseline_data["average"]["full_curve_mean"])
        epochs = range(1, len(base_mean) + 1)
        
                         
        if "runs" in baseline_data:
            label_added = False
            for run in baseline_data["runs"]:
                if "full_curve" in run:
                    data = run["full_curve"]
                    if len(data) > 0:
                        label = 'Baseline Runs' if not label_added else ""
                        plt.plot(range(1, len(data) + 1), data, color='#1f77b4', alpha=0.4, linewidth=1, linestyle=':', label=label)
                        label_added = True
        
              
        plt.plot(epochs, base_mean, 'o--', color='#1f77b4', label='Baseline Mean', linewidth=2.5, markersize=6)
    
                             
    if "average" in freezing_data and "full_curve_mean" in freezing_data["average"]:
        freez_mean = np.array(freezing_data["average"]["full_curve_mean"])
        epochs = range(1, len(freez_mean) + 1)
        
                         
        if "runs" in freezing_data:
            label_added = False
            for run in freezing_data["runs"]:
                if "full_curve" in run:
                    data = run["full_curve"]
                    if len(data) > 0:
                        label = 'Freezing Runs' if not label_added else ""
                        plt.plot(range(1, len(data) + 1), data, color='#D62728', alpha=0.4, linewidth=1, linestyle='-', label=label)
                        label_added = True
        
              
        plt.plot(epochs, freez_mean, 'o-', color='#D62728', label='Freezing Mean', linewidth=3, markersize=8)

                        
    plt.title('Task A Accuracy (Catastrophic Forgetting)', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.ylim(0, 105)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    
                        
    boundary = 5.5
    plt.axvline(x=boundary, color='black', linestyle='--', alpha=0.5, label='Task Boundary')
    plt.text(boundary/2, 5, 'Phase 1: Task A', ha='center', fontsize=10, alpha=0.7)
    plt.text(boundary + boundary/2, 5, 'Phase 2: Task B', ha='center', fontsize=10, alpha=0.7)

    plt.savefig('task_a_accuracy.png')
    print("[Success] Saved 'task_a_accuracy.png'")
    plt.close()

                                                      
    plt.figure(figsize=(10, 6), dpi=150)
    
                              
    if "average" in baseline_data and "task_b_mean" in baseline_data["average"]:
        base_task_b_mean = np.array(baseline_data["average"]["task_b_mean"])
        epochs_b = range(6, 6 + len(base_task_b_mean))
        
                         
        if "runs" in baseline_data:
            label_added = False
            for run in baseline_data["runs"]:
                if "task_b" in run:
                    data = run["task_b"]
                    if len(data) > 0:
                                                           
                        label = 'Baseline Runs' if not label_added else ""
                        plt.plot(range(6, 6 + len(data)), data, color='#1f77b4', alpha=0.4, linewidth=1, linestyle=':', label=label)
                        label_added = True
        
              
        plt.plot(epochs_b, base_task_b_mean, 'o--', color='#1f77b4', label='Baseline Mean', linewidth=2.5, markersize=6)

                             
    if "average" in freezing_data and "task_b_mean" in freezing_data["average"]:
        task_b_mean = np.array(freezing_data["average"]["task_b_mean"])
        epochs_b = range(6, 6 + len(task_b_mean))
        
                         
        if "runs" in freezing_data:
            label_added = False
            for run in freezing_data["runs"]:
                if "task_b" in run:
                    data = run["task_b"]
                    if len(data) > 0:
                        label = 'Freezing Runs' if not label_added else ""
                        plt.plot(range(6, 6 + len(data)), data, color='#D62728', alpha=0.4, linewidth=1, linestyle='-', label=label)
                        label_added = True
        
              
        plt.plot(epochs_b, task_b_mean, 'o-', color='#D62728', label='Freezing Mean', linewidth=3, markersize=8)

                        
    plt.title('Task B Accuracy (Learning Curve)', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.ylim(0, 105)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.xticks(range(6, 11))

    plt.savefig('task_b_accuracy.png')
    print("[Success] Saved 'task_b_accuracy.png'")
    plt.close()

import glob

def plot_percentile_comparison(results_dir):
                                       
    try:
        dir_name = os.path.basename(os.path.normpath(results_dir))
        epochs = dir_name.split('_')[-1]
    except:
        epochs = "unknown"

                   
    baseline_path = os.path.join(results_dir, "cl_baseline.json")
    baseline_a = None
    baseline_b = None
    
    if os.path.exists(baseline_path):
        with open(baseline_path, "r") as f:
            base_data = json.load(f)
            if "average" in base_data:
                baseline_a = base_data["average"].get("final_task_a_mean")
                                             
                tb = base_data["average"].get("task_b_mean")
                if isinstance(tb, list) and tb:
                    baseline_b = tb[-1]

                        
    freezing_files = glob.glob(os.path.join(results_dir, "freezing_*.json"))
    percentiles = []
    task_a_means = []
    task_a_stds = []
    task_b_means = []
    task_b_stds = []
    
    for f in freezing_files:
        try:
            filename = os.path.basename(f)
            p_str = filename.replace("freezing_", "").replace(".json", "")
            p = float(p_str) / 100.0
            
            with open(f, "r") as fp:
                data = json.load(fp)
                if "average" in data:
                    avg = data["average"]
                    
                    percentiles.append(p)
                    task_a_means.append(avg.get("final_task_a_mean", 0))
                    task_a_stds.append(avg.get("final_task_a_std", 0))
                    
                    tb = avg.get("task_b_mean", [])
                    tbs = avg.get("task_b_std", [])
                    task_b_means.append(tb[-1] if isinstance(tb, list) and tb else 0)
                    task_b_stds.append(tbs[-1] if isinstance(tbs, list) and tbs else 0)
        except Exception as e:
            print(f"Error processing {f}: {e}")

                        
    if not percentiles:
        print("No freezing data found for plotting.")
        return

    sorted_indices = np.argsort(percentiles)
    percentiles = np.array(percentiles)[sorted_indices]
    task_a_means = np.array(task_a_means)[sorted_indices]
    task_a_stds = np.array(task_a_stds)[sorted_indices]
    task_b_means = np.array(task_b_means)[sorted_indices]
    task_b_stds = np.array(task_b_stds)[sorted_indices]

                                                    
    plt.figure(figsize=(10, 6), dpi=150)
    plt.errorbar(percentiles, task_a_means, yerr=task_a_stds, fmt='o-', color='#D62728', label='Freezing (Task A)', capsize=5)
    
    if baseline_a is not None:
        plt.axhline(y=baseline_a, color='#1f77b4', linestyle='--', label=f'Baseline Task A ({baseline_a:.1f}%)')
        
    plt.title('Task A Retention vs Freezing Percentile', fontsize=14, fontweight='bold')
    plt.xlabel('Freezing Percentile', fontsize=12)
    plt.ylabel('Retention Accuracy (%)', fontsize=12)
    plt.ylim(0, 105)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    
    save_path_a = os.path.join(results_dir, f'{epochs}_epoch_percentile_vs_task_a.png')
    plt.savefig(save_path_a)
    print(f"[Success] Saved '{save_path_a}'")
    plt.close()

                                                   
    plt.figure(figsize=(10, 6), dpi=150)
    plt.errorbar(percentiles, task_b_means, yerr=task_b_stds, fmt='o-', color='#2CA02C', label='Freezing (Task B)', capsize=5)
    
    if baseline_b is not None:
        plt.axhline(y=baseline_b, color='#1f77b4', linestyle='--', label=f'Baseline Task B ({baseline_b:.1f}%)')
        
    plt.title('Task B Accuracy vs Freezing Percentile', fontsize=14, fontweight='bold')
    plt.xlabel('Freezing Percentile', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.ylim(0, 105)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    
    save_path_b = os.path.join(results_dir, f'{epochs}_epoch_percentile_vs_task_b.png')
    plt.savefig(save_path_b)
    print(f"[Success] Saved '{save_path_b}'")
    plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results_epochs_1", help="Directory containing results")
    args = parser.parse_args()
    
                                                                      
                                                                  
                                                                                                     
                                        
    
    plot_percentile_comparison(args.results_dir)
