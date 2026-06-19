import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'axes.axisbelow': True,
    'axes.spines.top': False,
    'axes.spines.right': False
})

def plot_ablations():
    # Data
    methods = ['No-Scale', 'Reset-Zero', 'P-Factor (Default)']
    
    # Task A Class-IL
    task_a = [30.36, 46.77, 64.46]
    task_a_err = [13.34, 15.09, 8.99]
    
    # Task B Class-IL
    task_b = [94.33, 84.54, 88.59]
    task_b_err = [1.82, 12.07, 5.35]
    
    # Combined
    combined = [63.25, 69.51, 79.88]
    combined_err = [0, 0, 0]

    x = np.arange(len(methods))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))

    rects1 = ax.bar(x - width, task_a, width, yerr=task_a_err, capsize=4, label='Task A (Class-IL)', color='#3498db', edgecolor='black', alpha=0.9)
    rects2 = ax.bar(x, task_b, width, yerr=task_b_err, capsize=4, label='Task B (Class-IL)', color='#e67e22', edgecolor='black', alpha=0.9)
    rects3 = ax.bar(x + width, combined, width, capsize=4, label='Combined Accuracy', color='#9b59b6', edgecolor='black', alpha=0.9)

    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_title('Ablation Study: Scaling & Reset Mechanics (Epoch 3)', fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(0, 105)

    ax.legend(loc='lower right', frameon=True, framealpha=0.9)

    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    out_path = 'plots/ablations_comparison.pdf'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    plot_ablations()
