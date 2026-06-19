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

def plot_progression():
    # Data extraction from the table
    # Columns: After Task 1, 2, 3, 4, 5
    tasks_x = [1, 2, 3, 4, 5]
    
    # Class-IL data
    cil_t1 = [83.1, 22.0, 24.4, 19.9, 20.9]
    cil_t2 = [np.nan, 76.6, 1.4, 18.9, 15.3]
    cil_t3 = [np.nan, np.nan, 91.2, 78.4, 78.4]
    cil_t4 = [np.nan, np.nan, np.nan, 19.9, 9.1]
    cil_t5 = [np.nan, np.nan, np.nan, np.nan, 18.7]
    
    # Task-IL data
    til_t1 = [88.9, 79.9, 78.5, 84.2, 83.5]
    til_t2 = [np.nan, 84.4, 84.5, 76.9, 76.8]
    til_t3 = [np.nan, np.nan, 99.8, 96.3, 94.5]
    til_t4 = [np.nan, np.nan, np.nan, 60.0, 60.0]
    til_t5 = [np.nan, np.nan, np.nan, np.nan, 60.2]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    markers = ['o', 's', '^', 'D', 'v']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Plot Class-IL
    ax1.plot(tasks_x, cil_t1, marker=markers[0], color=colors[0], linewidth=2.5, markersize=8, label='Eval Task 1')
    ax1.plot(tasks_x, cil_t2, marker=markers[1], color=colors[1], linewidth=2.5, markersize=8, label='Eval Task 2')
    ax1.plot(tasks_x, cil_t3, marker=markers[2], color=colors[2], linewidth=2.5, markersize=8, label='Eval Task 3')
    ax1.plot(tasks_x, cil_t4, marker=markers[3], color=colors[3], linewidth=2.5, markersize=8, label='Eval Task 4')
    ax1.plot(tasks_x, cil_t5, marker=markers[4], color=colors[4], linewidth=2.5, markersize=8, label='Eval Task 5')
    
    ax1.set_title('P-Factor Class-IL Trajectory', fontweight='bold', pad=15)
    ax1.set_xlabel('Tasks Encountered', fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontweight='bold')
    ax1.set_xticks(tasks_x)
    ax1.set_ylim(-5, 105)
    
    # Plot Task-IL
    ax2.plot(tasks_x, til_t1, marker=markers[0], color=colors[0], linewidth=2.5, markersize=8, label='Eval Task 1')
    ax2.plot(tasks_x, til_t2, marker=markers[1], color=colors[1], linewidth=2.5, markersize=8, label='Eval Task 2')
    ax2.plot(tasks_x, til_t3, marker=markers[2], color=colors[2], linewidth=2.5, markersize=8, label='Eval Task 3')
    ax2.plot(tasks_x, til_t4, marker=markers[3], color=colors[3], linewidth=2.5, markersize=8, label='Eval Task 4')
    ax2.plot(tasks_x, til_t5, marker=markers[4], color=colors[4], linewidth=2.5, markersize=8, label='Eval Task 5')
    
    ax2.set_title('P-Factor Task-IL Trajectory', fontweight='bold', pad=15)
    ax2.set_xlabel('Tasks Encountered', fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontweight='bold')
    ax2.set_xticks(tasks_x)
    ax2.set_ylim(-5, 105)
    
    # Common Legend
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.05), frameon=False)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    
    os.makedirs('plots', exist_ok=True)
    out_path = 'plots/fashion_progression.pdf'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    plot_progression()
