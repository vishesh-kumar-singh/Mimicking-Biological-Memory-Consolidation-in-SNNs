import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
import glob
import re
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

                               
sys.path.append(os.getcwd())

from src.generate_summary import generate_summary, generate_root_readme

                          

def load_json(filepath):
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def get_all_results(epochs_list=[1, 2, 3, 4, 5]):
    """
    Loads all results from results_epochs_X directories.
    Returns a dictionary structure:
    {
        epoch: {
            'baseline': data,
            'freezing': { percentile: data, ... }
        }
    }
    """
    all_results = {}
    for epoch in epochs_list:
        results_dir = f"results/SNN/Split-MNIST/epochs_{epoch}"
        if not os.path.exists(results_dir):
            continue
            
        all_results[epoch] = {'freezing': {}}
        
                       
        baseline_path = os.path.join(results_dir, "cl_baseline.json")
        if os.path.exists(baseline_path):
            all_results[epoch]['baseline'] = load_json(baseline_path)
            
                       
        freezing_files = glob.glob(os.path.join(results_dir, "freezing_*.json"))
        for f in freezing_files:
            try:
                filename = os.path.basename(f)
                p_str = filename.replace("freezing_", "").replace(".json", "")
                percentile = float(p_str) / 100.0
                all_results[epoch]['freezing'][percentile] = load_json(f)
            except:
                pass
                
    return all_results

                              

def plot_baseline_vs_pfactor_stability(all_results):
    """
    Generates the 'Baseline Failure vs. P-Factor Stability' plot.
    X-Axis: Epochs (1 to 5)
    Y-Axis: Task A Retention (%)
    Line 1 (Red): Baseline
    Line 2 (Blue): Tau=0.8 Average
    Line 3 (Green): Tau=0.8 Best Seed
    """
    print("Generating Paper Plot: Baseline Failure vs. P-Factor Stability...")
    
    epochs = sorted(all_results.keys())
    baseline_means = []
    p80_means = []
    p80_bests = []
    
    for e in epochs:
        data = all_results[e]
        
                  
        base_val = 0
        if 'baseline' in data and data['baseline'] and "average" in data['baseline']:
            base_val = data['baseline']["average"].get("final_task_a_mean", 0)
        baseline_means.append(base_val)
        
                         
        p80_avg = 0
        p80_best = 0
        if 'freezing' in data and 0.8 in data['freezing']:
            res = data['freezing'][0.8]
            if res and "average" in res:
                p80_avg = res["average"].get("final_task_a_mean", 0)
            
                                    
            if res and "runs" in res:
                best_run_val = -1
                for run in res["runs"]:
                    val = run.get("final_task_a", 0)
                    if val > best_run_val:
                        best_run_val = val
                p80_best = best_run_val if best_run_val != -1 else 0
                
        p80_means.append(p80_avg)
        p80_bests.append(p80_best)
        
    plt.figure(figsize=(10, 6), dpi=150)
    
                
    plt.plot(epochs, baseline_means, 'o-', color='red', label='Baseline', linewidth=2, markersize=8)
    plt.plot(epochs, p80_means, 's--', color='blue', label=r'$\tau=0.8$ Average', linewidth=2, markersize=8)
    plt.plot(epochs, p80_bests, '^-.', color='green', label=r'$\tau=0.8$ Best Seed', linewidth=2, markersize=8)
    
    plt.title('Longitudinal Comparison of Memory Retention', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Task A Retention (%)', fontsize=12)
    plt.ylim(-5, 105)
    plt.xticks(epochs)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=11)
    
    os.makedirs("plots", exist_ok=True)
    plt.savefig('plots/baseline_vs_pfactor.png')
    plt.close()
    print("Saved plots/baseline_vs_pfactor.png")

def plot_epoch3_tradeoff_curve(all_results):
    """
    Generates the 'Epoch 3 Trade-off Curve'.
    X-Axis: Percentile (0.4 to 0.8)
    Y-Axis: Accuracy (%)
    Curve 1: Task A Retention (Average)
    Curve 2: Task B Accuracy (Average)
    Curve 3: Task A Retention (Best Seed)
    Curve 4: Task B Accuracy (Best Seed)
    Highlight peak retention.
    """
    print("Generating Paper Plot: Epoch 3 Trade-off Curve...")
    
    if 3 not in all_results:
        print("No data for Epoch 3.")
        return
        
    data = all_results[3]['freezing']
    percentiles = sorted([p for p in data.keys() if p >= 0.4])
    
    task_a_means = []
    task_b_means = []
    task_a_bests = []
    task_b_bests = []
    
    best_retention = -1
    best_p = -1
    best_coords = (0, 0)
    
    for p in percentiles:
        res = data[p]
        if res:
                      
            a_mean = 0
            b_mean = 0
            if "average" in res:
                a_mean = res["average"].get("final_task_a_mean", 0)
                b_list = res["average"].get("task_b_mean", [])
                b_mean = b_list[-1] if b_list else 0
            
            task_a_means.append(a_mean)
            task_b_means.append(b_mean)
            
                                                   
            a_best = 0
            b_best = 0
            if "runs" in res:
                best_run_val = -1
                best_run_idx = -1
                for idx, run in enumerate(res["runs"]):
                    val = run.get("final_task_a", 0)
                    if val > best_run_val:
                        best_run_val = val
                        best_run_idx = idx
                
                if best_run_idx != -1:
                    a_best = best_run_val
                    b_list = res["runs"][best_run_idx].get("task_b", [])
                    b_best = b_list[-1] if b_list else 0
            
            task_a_bests.append(a_best)
            task_b_bests.append(b_best)
            
                                                                      
            if a_best > best_retention:
                best_retention = a_best
                best_p = p
                best_coords = (p, a_best)
    
    plt.figure(figsize=(10, 6), dpi=150)
    
                   
    plt.plot(percentiles, task_a_means, 'o-', color='#D62728', label='Task A Retention (Avg)', linewidth=2, markersize=6, alpha=0.7)
    plt.plot(percentiles, task_b_means, 's-', color='#2CA02C', label='Task B Accuracy (Avg)', linewidth=2, markersize=6, alpha=0.7)
    
                     
    plt.plot(percentiles, task_a_bests, 'o--', color='#8B0000', label='Task A Retention (Best)', linewidth=2.5, markersize=8)
    plt.plot(percentiles, task_b_bests, 's--', color='#006400', label='Task B Accuracy (Best)', linewidth=2.5, markersize=8)
    
                    
    if best_p != -1:
        plt.annotate(f'Peak: {best_retention:.2f}%', 
                     xy=best_coords, 
                     xytext=(0, 15), textcoords='offset points',
                     ha='center', va='bottom',
                     bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                     arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
    
    plt.title('Stability-Plasticity Trade-off at Epoch 3', fontsize=14, fontweight='bold')
    plt.xlabel(r'Consolidation Threshold ($\tau$)', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.ylim(0, 105)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='center right', fontsize=10)
    
    os.makedirs("plots", exist_ok=True)
    plt.savefig('plots/epoch3_tradeoff.png')
    plt.close()
    print("Saved plots/epoch3_tradeoff.png")

                                             

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
        print(f"No freezing data found in {results_dir} for plotting.")
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
        
    plt.title(f'Task A Retention vs Freezing Percentile (Epochs: {epochs})', fontsize=14, fontweight='bold')
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
        
    plt.title(f'Task B Accuracy vs Freezing Percentile (Epochs: {epochs})', fontsize=14, fontweight='bold')
    plt.xlabel('Freezing Percentile', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.ylim(0, 105)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    
    save_path_b = os.path.join(results_dir, f'{epochs}_epoch_percentile_vs_task_b.png')
    plt.savefig(save_path_b)
    print(f"[Success] Saved '{save_path_b}'")
    plt.close()

def plot_detailed_curves(results_dir):
    """
    Plots detailed accuracy curves (Task A Forgetting, Task B Learning) for a specific directory.
    Adapted from plot_results.py's plot_results().
    """
               
    baseline_data = load_json(os.path.join(results_dir, "cl_baseline.json"))
    
                                                          
    freezing_path = os.path.join(results_dir, "freezing_70.json")
    if not os.path.exists(freezing_path):
                                          
        files = glob.glob(os.path.join(results_dir, "freezing_*.json"))
        if files:
            freezing_path = files[0]                          
        else:
            freezing_path = None
            
    freezing_data = load_json(freezing_path) if freezing_path else None
    
    if not baseline_data and not freezing_data:
        print(f"No data found in {results_dir} for detailed curves.")
        return

                        
    try:
        dir_name = os.path.basename(os.path.normpath(results_dir))
        epochs_per_task = int(dir_name.split('_')[-1])
    except:
        epochs_per_task = 5          

                                                        
    plt.figure(figsize=(10, 6), dpi=150)
    
                              
    if baseline_data and "average" in baseline_data and "full_curve_mean" in baseline_data["average"]:
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
    
                             
    if freezing_data and "average" in freezing_data and "full_curve_mean" in freezing_data["average"]:
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

                        
    plt.title(f'Task A Accuracy (Catastrophic Forgetting) - {epochs_per_task} Epochs', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.ylim(0, 105)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    
                        
    boundary = epochs_per_task + 0.5
    plt.axvline(x=boundary, color='black', linestyle='--', alpha=0.5, label='Task Boundary')
    
    save_path = os.path.join(results_dir, 'task_a_accuracy.png')
    plt.savefig(save_path)
    print(f"[Success] Saved '{save_path}'")
    plt.close()

                                                      
    plt.figure(figsize=(10, 6), dpi=150)
    
                              
    if baseline_data and "average" in baseline_data and "task_b_mean" in baseline_data["average"]:
        base_task_b_mean = np.array(baseline_data["average"]["task_b_mean"])
        epochs_b = range(epochs_per_task + 1, epochs_per_task + 1 + len(base_task_b_mean))
        
                         
        if "runs" in baseline_data:
            label_added = False
            for run in baseline_data["runs"]:
                if "task_b" in run:
                    data = run["task_b"]
                    if len(data) > 0:
                        label = 'Baseline Runs' if not label_added else ""
                        plt.plot(range(epochs_per_task + 1, epochs_per_task + 1 + len(data)), data, color='#1f77b4', alpha=0.4, linewidth=1, linestyle=':', label=label)
                        label_added = True
        
              
        plt.plot(epochs_b, base_task_b_mean, 'o--', color='#1f77b4', label='Baseline Mean', linewidth=2.5, markersize=6)

                             
    if freezing_data and "average" in freezing_data and "task_b_mean" in freezing_data["average"]:
        task_b_mean = np.array(freezing_data["average"]["task_b_mean"])
        epochs_b = range(epochs_per_task + 1, epochs_per_task + 1 + len(task_b_mean))
        
                         
        if "runs" in freezing_data:
            label_added = False
            for run in freezing_data["runs"]:
                if "task_b" in run:
                    data = run["task_b"]
                    if len(data) > 0:
                        label = 'Freezing Runs' if not label_added else ""
                        plt.plot(range(epochs_per_task + 1, epochs_per_task + 1 + len(data)), data, color='#D62728', alpha=0.4, linewidth=1, linestyle='-', label=label)
                        label_added = True
        
              
        plt.plot(epochs_b, task_b_mean, 'o-', color='#D62728', label='Freezing Mean', linewidth=3, markersize=8)

                        
    plt.title(f'Task B Accuracy (Learning Curve) - {epochs_per_task} Epochs', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.ylim(0, 105)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    
    save_path = os.path.join(results_dir, 'task_b_accuracy.png')
    plt.savefig(save_path)
    print(f"[Success] Saved '{save_path}'")
    plt.close()

                        

def analyze_all_folders():
                                           
    all_dirs = glob.glob("results/neuron_level/epochs_*")
    
    epoch_dirs = []
    for d in all_dirs:
        if os.path.isdir(d):
            match = re.search(r'epochs_(\d+)', d)
            if match:
                epochs = int(match.group(1))
                epoch_dirs.append((epochs, d))
    
    epoch_dirs.sort(key=lambda x: x[0])
    
    if not epoch_dirs:
        print("No neuron_level/epochs_* directories found.")
        return

    print(f"Found {len(epoch_dirs)} result directories: {[d[1] for d in epoch_dirs]}")
    
                                                        
    for epochs, results_dir in epoch_dirs:
        print(f"\n{'='*40}")
        print(f"Analyzing {results_dir} (Epochs: {epochs})")
        print(f"{'='*40}")
        
                                 
        print(f"Generating Summary for {results_dir}...")
        try:
            generate_summary(results_dir)
        except Exception as e:
            print(f"Error generating summary for {results_dir}: {e}")
            
                                                       
        print(f"Generating Percentile Comparison Plots for {results_dir}...")
        try:
            plot_percentile_comparison(results_dir)
        except Exception as e:
            print(f"Error generating percentile plots for {results_dir}: {e}")
            
                                                         
        print(f"Generating Detailed Curve Plots for {results_dir}...")
        try:
            plot_detailed_curves(results_dir)
        except Exception as e:
            print(f"Error generating detailed curves for {results_dir}: {e}")

                                               
    print(f"\n{'='*40}")
    print("Generating Global Paper Plots")
    print(f"{'='*40}")
    
    all_results = get_all_results([d[0] for d in epoch_dirs])
    
    try:
        plot_baseline_vs_pfactor_stability(all_results)
    except Exception as e:
        print(f"Error generating baseline vs pfactor plot: {e}")
        
    try:
        plot_epoch3_tradeoff_curve(all_results)
    except Exception as e:
        print(f"Error generating epoch 3 tradeoff plot: {e}")

                             
    print("\nGenerating Root README...")
    try:
        generate_root_readme()
    except Exception as e:
        print(f"Error generating root README: {e}")

if __name__ == "__main__":
    analyze_all_folders()
