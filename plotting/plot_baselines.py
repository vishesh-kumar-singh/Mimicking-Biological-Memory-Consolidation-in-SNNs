import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 11,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'axes.axisbelow': True,
    'axes.spines.top': False,
    'axes.spines.right': False
})

def plot_baselines():
    # Full Table 1 Data + P-Factor (Ours)
    methods = [
        'Fine-Tune',
        r'EWC($10^3$)', r'EWC($10^5$)', r'EWC($10^6$)',
        r'SI($1$)', r'SI($10^2$)', r'SI($10^4$)', r'SI($10^6$)',
        'P-Factor'
    ]
    
    # Task A Class-IL
    task_a = [0.0, 0.59, 15.89, 23.62, 7.60, 18.55, 27.30, 54.97, 64.46]
    task_a_err = [0.0, 0.93, 7.48, 3.63, 4.22, 8.29, 6.73, 11.02, 8.99]
    
    # Task B Class-IL
    task_b = [95.60, 95.51, 34.10, 4.11, 96.11, 94.53, 81.96, 66.47, 88.59]
    task_b_err = [1.02, 0.60, 36.46, 8.21, 0.99, 0.91, 14.39, 17.01, 5.35]
    
    # Combined
    combined = [48.09, 48.25, 32.51, 15.54, 51.85, 56.49, 60.92, 68.18, 79.88]

    x = np.arange(len(methods))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    rects1 = ax.bar(x - width, task_a, width, yerr=task_a_err, capsize=3, label='Task A (Class-IL)', color='#3498db', edgecolor='black', alpha=0.9)
    rects2 = ax.bar(x, task_b, width, yerr=task_b_err, capsize=3, label='Task B (Class-IL)', color='#e67e22', edgecolor='black', alpha=0.9)
    rects3 = ax.bar(x + width, combined, width, capsize=3, label='Combined Accuracy', color='#9b59b6', edgecolor='black', alpha=0.9)

    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_title('Gradient-Based Baselines vs. P-Factor (Split-MNIST, Epoch 3)', fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.set_ylim(0, 105)

    ax.legend(loc='upper right', frameon=True, framealpha=0.9)

    # Add a vertical line to separate baselines from ours
    ax.axvline(x=7.5, color='black', linestyle='-.', alpha=0.5)

    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    out_path = 'plots/baselines_comparison.pdf'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    plot_baselines()
