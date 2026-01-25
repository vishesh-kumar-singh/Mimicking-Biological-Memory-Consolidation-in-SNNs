import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import argparse

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
        results_dir = f"results_epochs_{epoch}"
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

def plot_accuracy_vs_epochs(all_results, target_epoch=5):
    """
    Plots accuracy curves for a specific training duration (e.g., 5 epochs per task).
    """
    print(f"Generating Accuracy vs. Epochs plots for {target_epoch} epochs/task...")
    
    if target_epoch not in all_results:
        print(f"No data for {target_epoch} epochs.")
        return

    data = all_results[target_epoch]
    baseline = data.get('baseline')
    freezing = data.get('freezing', {})
    
                                                        
    best_p = None
    best_retention = -1
    for p, res in freezing.items():
        if res and "average" in res:
            ret = res["average"].get("final_task_a_mean", 0)
            if ret > best_retention:
                best_retention = ret
                best_p = p
                
                                      
    plt.figure(figsize=(10, 6), dpi=150)
    
              
    if baseline and "average" in baseline:
        mean = baseline["average"].get("full_curve_mean", [])
        if mean:
            epochs = range(1, len(mean) + 1)
            plt.plot(epochs, mean, 'o--', color='#1f77b4', label='Baseline', linewidth=2)
            
                   
    if best_p is not None:
        res = freezing[best_p]
        mean = res["average"].get("full_curve_mean", [])
        if mean:
            epochs = range(1, len(mean) + 1)
            plt.plot(epochs, mean, 'o-', color='#D62728', label=f'Freezing (Top {best_p*100:.0f}%)', linewidth=2)
            
    plt.axvline(x=target_epoch + 0.5, color='gray', linestyle=':', alpha=0.5, label='Task Boundary')
    plt.title(f'Task A Accuracy (Forgetting) - {target_epoch} Epochs/Task', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.ylim(0, 105)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(f'plots/accuracy_task_a_epochs_{target_epoch}.png')
    plt.close()
    
                                    
    plt.figure(figsize=(10, 6), dpi=150)
    
              
    if baseline and "average" in baseline:
        mean = baseline["average"].get("task_b_mean", [])
        if mean:
            epochs = range(target_epoch + 1, target_epoch + 1 + len(mean))
            plt.plot(epochs, mean, 'o--', color='#1f77b4', label='Baseline', linewidth=2)

                   
    if best_p is not None:
        res = freezing[best_p]
        mean = res["average"].get("task_b_mean", [])
        if mean:
            epochs = range(target_epoch + 1, target_epoch + 1 + len(mean))
            plt.plot(epochs, mean, 'o-', color='#D62728', label=f'Freezing (Top {best_p*100:.0f}%)', linewidth=2)

    plt.title(f'Task B Accuracy (Learning) - {target_epoch} Epochs/Task', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.ylim(0, 105)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(f'plots/accuracy_task_b_epochs_{target_epoch}.png')
    plt.close()

def plot_retention_vs_plasticity(all_results):
    """
    Scatter plot of Final Task A vs. Final Task B accuracy for all experiments.
    """
    print("Generating Retention vs. Plasticity trade-off plot...")
    
    plt.figure(figsize=(10, 8), dpi=150)
    
    colors = {1: 'red', 2: 'orange', 3: 'green', 4: 'blue', 5: 'purple'}
    markers = {1: 'o', 2: 's', 3: '^', 4: 'D', 5: 'v'}
    
    for epoch, data in all_results.items():
        c = colors.get(epoch, 'black')
        m = markers.get(epoch, 'o')
        
                  
        baseline = data.get('baseline')
        if baseline and "average" in baseline:
            a = baseline["average"].get("final_task_a_mean", 0)
            b_list = baseline["average"].get("task_b_mean", [])
            b = b_list[-1] if b_list else 0
            plt.scatter(b, a, color=c, marker=m, facecolors='none', edgecolors=c, s=100, label=f'Baseline (E={epoch})' if epoch==1 else "")
            
                  
        freezing = data.get('freezing', {})
        for p, res in freezing.items():
            if res and "average" in res:
                a = res["average"].get("final_task_a_mean", 0)
                b_list = res["average"].get("task_b_mean", [])
                b = b_list[-1] if b_list else 0
                plt.scatter(b, a, color=c, marker=m, s=50, alpha=0.7, label=f'Freezing (E={epoch})' if p==0.8 else "")                  
                
    plt.title('Stability-Plasticity Trade-off', fontsize=16)
    plt.xlabel('Plasticity (Task B Accuracy %)', fontsize=14)
    plt.ylabel('Stability (Task A Retention %)', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xlim(0, 105)
    plt.ylim(0, 105)
    
                                          
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Baseline', markerfacecolor='none', markeredgecolor='black', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Freezing', markerfacecolor='black', markersize=8),
        Line2D([0], [0], color='red', lw=2, label='1 Epoch'),
        Line2D([0], [0], color='orange', lw=2, label='2 Epochs'),
        Line2D([0], [0], color='green', lw=2, label='3 Epochs'),
        Line2D([0], [0], color='blue', lw=2, label='4 Epochs'),
        Line2D([0], [0], color='purple', lw=2, label='5 Epochs')
    ]
    plt.legend(handles=legend_elements, loc='lower left')
    
    plt.savefig('plots/tradeoff_scatter.png')
    plt.close()

def generate_inferences(all_results):
    print("\n" + "="*40)
    print("EXPERIMENTAL INFERENCES")
    print("="*40)
    
    for epoch in sorted(all_results.keys()):
        print(f"\n--- {epoch} Epochs per Task ---")
        data = all_results[epoch]
        baseline = data.get('baseline')
        freezing = data.get('freezing', {})
        
                        
        base_a = 0
        base_b = 0
        if baseline and "average" in baseline:
            base_a = baseline["average"].get("final_task_a_mean", 0)
            b_list = baseline["average"].get("task_b_mean", [])
            base_b = b_list[-1] if b_list else 0
            print(f"Baseline: Task A Retention = {base_a:.2f}%, Task B Accuracy = {base_b:.2f}%")
            print(f"  -> Catastrophic forgetting is {'SEVERE' if base_a < 10 else 'MODERATE' if base_a < 50 else 'MILD'}.")
            
                        
        best_p = None
        best_a = -1
        best_b = -1
        
        for p, res in freezing.items():
            if res and "average" in res:
                a = res["average"].get("final_task_a_mean", 0)
                b_list = res["average"].get("task_b_mean", [])
                b = b_list[-1] if b_list else 0
                
                if a > best_a:
                    best_a = a
                    best_b = b
                    best_p = p
                    
        if best_p is not None:
            print(f"Best Freezing ({best_p*100:.0f}%): Task A Retention = {best_a:.2f}%, Task B Accuracy = {best_b:.2f}%")
            improvement = best_a - base_a
            print(f"  -> Retention improved by {improvement:.2f}% compared to baseline.")
            
            if best_b < base_b - 5:
                print(f"  -> WARNING: Significant drop in Task B accuracy (-{base_b - best_b:.2f}%). Plasticity compromised.")
            elif best_b > base_b:
                print(f"  -> Forward transfer or better optimization observed (+{best_b - base_b:.2f}% Task B).")
            else:
                print(f"  -> Task B accuracy maintained within acceptable range.")

def main():
    os.makedirs("plots", exist_ok=True)
    
    all_results = get_all_results()
    
                    
    for epoch in all_results.keys():
        plot_accuracy_vs_epochs(all_results, target_epoch=epoch)
        
    plot_retention_vs_plasticity(all_results)
    
                              
    generate_inferences(all_results)
    
                                   
    plot_baseline_vs_pfactor_stability(all_results)
    plot_epoch3_tradeoff_curve(all_results)

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
    
    plt.savefig('plots/epoch3_tradeoff.png')
    plt.close()
    print("Saved plots/epoch3_tradeoff.png")

if __name__ == "__main__":
    main()
