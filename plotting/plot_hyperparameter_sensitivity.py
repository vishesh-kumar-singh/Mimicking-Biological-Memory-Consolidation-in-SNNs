import json
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12
})

def plot_sensitivity():
    # Load JSON
    json_path = 'results/SNN/hyperparameter_sensitivity.json'
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        return
        
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    # Extract unique alphas
    ltp_vals = sorted(list(set(d['ltp'] for d in data)))
    ltd_vals = sorted(list(set(d['ltd'] for d in data)))
    
    # Create empty matrices
    task_a_til_matrix = np.zeros((len(ltd_vals), len(ltp_vals)))
    task_b_cil_matrix = np.zeros((len(ltd_vals), len(ltp_vals)))
    combined_matrix = np.zeros((len(ltd_vals), len(ltp_vals)))
    
    # Fill matrices
    for d in data:
        ltp_idx = ltp_vals.index(d['ltp'])
        ltd_idx = ltd_vals.index(d['ltd'])
        
        task_a_til_matrix[ltd_idx, ltp_idx] = d['task_a_task_il']
        task_b_cil_matrix[ltd_idx, ltp_idx] = d['task_b_class_il']
        combined_matrix[ltd_idx, ltp_idx] = d['combined_acc']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Task A Task-IL
    sns.heatmap(task_a_til_matrix, annot=True, fmt=".1f", cmap="YlGnBu", 
                xticklabels=ltp_vals, yticklabels=ltd_vals, ax=axes[0],
                cbar_kws={'label': 'Accuracy (%)'})
    axes[0].set_title('Task A (Task-IL) Retention')
    axes[0].set_xlabel(r'$\alpha_{LTP}$')
    axes[0].set_ylabel(r'$\alpha_{LTD}$')
    
    # Plot 2: Task B Class-IL
    sns.heatmap(task_b_cil_matrix, annot=True, fmt=".1f", cmap="YlOrRd", 
                xticklabels=ltp_vals, yticklabels=ltd_vals, ax=axes[1],
                cbar_kws={'label': 'Accuracy (%)'})
    axes[1].set_title('Task B (Class-IL) Learning')
    axes[1].set_xlabel(r'$\alpha_{LTP}$')
    axes[1].set_ylabel(r'$\alpha_{LTD}$')
    
    # Plot 3: Combined Accuracy
    sns.heatmap(combined_matrix, annot=True, fmt=".1f", cmap="Purples", 
                xticklabels=ltp_vals, yticklabels=ltd_vals, ax=axes[2],
                cbar_kws={'label': 'Accuracy (%)'})
    axes[2].set_title('Combined Accuracy')
    axes[2].set_xlabel(r'$\alpha_{LTP}$')
    axes[2].set_ylabel(r'$\alpha_{LTD}$')
    
    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    out_path = 'plots/alpha_sensitivity_heatmaps.pdf'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved sensitivity plot to {out_path}")

if __name__ == "__main__":
    plot_sensitivity()
