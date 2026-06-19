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

def plot_structural_comparison():
    # Data extraction
    thresholds = [0.2, 0.4, 0.6, 0.8]
    x_labels = [r'$\tau=0.2$', r'$\tau=0.4$', r'$\tau=0.6$', r'$\tau=0.8$']
    
    # Task A Class-IL
    class_il_packnet = [1.02, 8.99, 16.22, 34.27]
    class_il_packnet_err = [0.87, 1.15, 2.37, 5.06]
    
    class_il_random = [3.08, 8.62, 22.97, 42.47]
    class_il_random_err = [2.92, 3.35, 7.91, 2.95]
    
    class_il_pfactor = [7.16, 17.36, 47.40, 64.46]
    class_il_pfactor_err = [6.92, 9.70, 7.12, 8.99]

    # Task A Task-IL
    task_il_packnet = [78.53, 96.49, 97.89, 98.37]
    task_il_packnet_err = [7.76, 0.78, 0.29, 0.18]
    
    task_il_random = [88.97, 95.16, 97.20, 98.14]
    task_il_random_err = [4.44, 1.52, 0.55, 0.27]
    
    task_il_pfactor = [62.39, 81.37, 94.96, 96.89]
    task_il_pfactor_err = [16.22, 9.73, 1.20, 1.62]
    
    # Combined Acc
    combined_packnet = [48.57, 52.79, 56.52, 65.86]
    combined_random = [49.62, 52.40, 59.79, 69.85]
    combined_pfactor = [51.58, 56.85, 72.42, 79.88]

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
    ax1.set_ylim(0, 100)
    
    # --- Subplot 2: Task A Task-IL ---
    ax2.bar(x - width, task_il_packnet, width, yerr=task_il_packnet_err, capsize=5, color=c_packnet, alpha=0.9, label='PackNet (Magnitude)', edgecolor='black')
    ax2.bar(x, task_il_random, width, yerr=task_il_random_err, capsize=5, color=c_random, alpha=0.9, label='Random Freezing', edgecolor='black')
    ax2.bar(x + width, task_il_pfactor, width, yerr=task_il_pfactor_err, capsize=5, color=c_pfactor, alpha=0.9, label='P-Factor (Ours)', edgecolor='black')
    
    ax2.set_ylabel('Task A Retention (%)', fontweight='bold')
    ax2.set_title('Task-IL Performance', fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(x_labels)
    ax2.set_ylim(40, 105) # Start at 40 to show variation better
    
    # Common Legend
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.05), frameon=False)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18) # Make room for legend
    
    # Save the figure
    os.makedirs('plots', exist_ok=True)
    out_path = 'plots/structural_comparison.pdf'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    plot_structural_comparison()
