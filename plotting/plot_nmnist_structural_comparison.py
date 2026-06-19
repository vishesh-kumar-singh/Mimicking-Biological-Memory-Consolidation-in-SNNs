import matplotlib.pyplot as plt
import numpy as np
import os

# Set global aesthetic parameters
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'axes.axisbelow': True,
    'axes.spines.top': False,
    'axes.spines.right': False
})

def plot_nmnist_structural_comparison():
    # Data extraction
    thresholds = [0.2, 0.4, 0.6, 0.8]
    x_labels = [r'$\tau=0.2$', r'$\tau=0.4$', r'$\tau=0.6$', r'$\tau=0.8$']
    
    # Task A Class-IL
    class_il_packnet = [0.69, 2.18, 3.66, 6.92]
    class_il_packnet_err = [0.70, 2.41, 3.01, 1.58]
    
    class_il_random = [0.51, 0.78, 2.20, 6.12]
    class_il_random_err = [0.23, 0.29, 1.35, 2.61]
    
    class_il_pfactor = [0.06, 0.04, 0.76, 4.73]
    class_il_pfactor_err = [0.11, 0.02, 0.72, 4.40]

    # Task A Task-IL
    task_il_packnet = [67.72, 92.61, 96.61, 98.25]
    task_il_packnet_err = [6.28, 2.17, 0.76, 0.34]
    
    task_il_random = [82.93, 91.79, 96.73, 98.09]
    task_il_random_err = [5.23, 4.91, 1.06, 0.30]
    
    task_il_pfactor = [80.46, 79.41, 93.95, 97.61]
    task_il_pfactor_err = [3.12, 4.94, 2.35, 0.89]

    # Colors and styles
    c_packnet = '#e74c3c' # Red
    c_random = '#95a5a6' # Grey
    c_pfactor = '#2ecc71' # Green
    
    x = np.arange(len(thresholds))
    width = 0.25

    # Create figure with 1x2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Subplot 1: Task A Class-IL ---
    ax1.bar(x - width, class_il_packnet, width, yerr=class_il_packnet_err, capsize=5, color=c_packnet, alpha=0.9, label='PackNet (Magnitude)', edgecolor='black')
    ax1.bar(x, class_il_random, width, yerr=class_il_random_err, capsize=5, color=c_random, alpha=0.9, label='Random Freezing', edgecolor='black')
    ax1.bar(x + width, class_il_pfactor, width, yerr=class_il_pfactor_err, capsize=5, color=c_pfactor, alpha=0.9, label='P-Factor (Ours)', edgecolor='black')
    
    ax1.set_ylabel('Task A Retention (%)', fontweight='bold')
    ax1.set_title('Class-IL Performance', fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels)
    ax1.set_ylim(0, 15)
    
    # --- Subplot 2: Task A Task-IL ---
    ax2.bar(x - width, task_il_packnet, width, yerr=task_il_packnet_err, capsize=5, color=c_packnet, alpha=0.9, label='PackNet (Magnitude)', edgecolor='black')
    ax2.bar(x, task_il_random, width, yerr=task_il_random_err, capsize=5, color=c_random, alpha=0.9, label='Random Freezing', edgecolor='black')
    ax2.bar(x + width, task_il_pfactor, width, yerr=task_il_pfactor_err, capsize=5, color=c_pfactor, alpha=0.9, label='P-Factor (Ours)', edgecolor='black')
    
    ax2.set_ylabel('Task A Retention (%)', fontweight='bold')
    ax2.set_title('Task-IL Performance', fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(x_labels)
    ax2.set_ylim(40, 105)
    
    # Common Legend
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.05), frameon=False)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18) # Make room for legend
    
    # Save the figure
    os.makedirs('plots', exist_ok=True)
    out_path = 'plots/nmnist_structural_comparison.pdf'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    plot_nmnist_structural_comparison()
