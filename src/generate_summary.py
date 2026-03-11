"""
generate_summary.py - Results Aggregation and README Generation

Generates per-folder README.md files summarizing experiment results and
an aggregated root README.md across all epoch configurations.

Supported experiment types:
- Baseline (cl_baseline.json): No freezing control
- Freezing (freezing_*.json): P-factor based engram freezing
- Random (random_*.json): Random neuron freezing control
- Index (index_*.json): Index-based neuron freezing control
- NoScale (noscale_*.json): P-factor tracking without weight scaling ablation

Usage:
------
    Called by scripts/analyze_results.py, not typically run directly.
"""

import os
import json
import glob
import re
import numpy as np
import argparse


def load_json(filepath):
    """Load and parse a JSON results file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def generate_summary(results_dir):
    """
    Generate a README.md summarizing all experiment results in a directory.
    
    Reads all JSON result files, extracts mean/std statistics, and writes
    a formatted markdown table for each experiment type.
    
    Args:
        results_dir (str): Path to results_epochs_N directory
    """
    readme_path = os.path.join(results_dir, "README.md")
    
    # Load Baseline
    baseline_path = os.path.join(results_dir, "cl_baseline.json")
    baseline_data = None
    if os.path.exists(baseline_path):
        baseline_data = load_json(baseline_path)
    
    # Load Freezing Results
    freezing_files = glob.glob(os.path.join(results_dir, "freezing_*.json"))
    freezing_results = []
    
    for f in freezing_files:
        filename = os.path.basename(f)
        try:
            percentile_str = filename.replace("freezing_", "").replace(".json", "")
            percentile = float(percentile_str) / 100.0
            data = load_json(f)
            freezing_results.append((percentile, data))
        except ValueError:
            continue
            
    freezing_results.sort(key=lambda x: x[0])

    # Load Random Results
    random_files = glob.glob(os.path.join(results_dir, "random_*.json"))
    random_results = []
    
    for f in random_files:
        filename = os.path.basename(f)
        try:
            percentile_str = filename.replace("random_", "").replace(".json", "")
            percentile = float(percentile_str) / 100.0
            data = load_json(f)
            random_results.append((percentile, data))
        except ValueError:
            continue
            
    random_results.sort(key=lambda x: x[0])
    
    # Load Index Freezing Results
    index_files = glob.glob(os.path.join(results_dir, "index_*.json"))
    index_results = []
    
    for f in index_files:
        filename = os.path.basename(f)
        try:
            percentile_str = filename.replace("index_", "").replace(".json", "")
            percentile = float(percentile_str) / 100.0
            data = load_json(f)
            index_results.append((percentile, data))
        except ValueError:
            continue
            
    index_results.sort(key=lambda x: x[0])

    # Load NoScale Freezing Results
    noscale_files = glob.glob(os.path.join(results_dir, "noscale_*.json"))
    noscale_results = []
    
    for f in noscale_files:
        filename = os.path.basename(f)
        try:
            percentile_str = filename.replace("noscale_", "").replace(".json", "")
            percentile = float(percentile_str) / 100.0
            data = load_json(f)
            noscale_results.append((percentile, data))
        except ValueError:
            continue
            
    noscale_results.sort(key=lambda x: x[0])
    
    with open(readme_path, "w") as f:
        f.write(f"# Experiment Results Summary ({os.path.basename(results_dir)})\n\n")
        
        # Baseline Section
        if baseline_data and "average" in baseline_data:
            avg = baseline_data["average"]
            f.write("## Baseline Results\n\n")
            f.write("| Metric | Mean | Std Dev |\n")
            f.write("| :--- | :--- | :--- |\n")
            f.write(f"| Final Task A Retention | {avg.get('final_task_a_mean', 'N/A'):.2f}% | {avg.get('final_task_a_std', 'N/A'):.2f} |\n")
            
            task_b_mean = avg.get('task_b_mean', [])
            task_b_std = avg.get('task_b_std', [])
            val_b_mean = task_b_mean[-1] if isinstance(task_b_mean, list) and task_b_mean else "N/A"
            val_b_std = task_b_std[-1] if isinstance(task_b_std, list) and task_b_std else "N/A"
            
            if isinstance(val_b_mean, (int, float)):
                f.write(f"| Final Task B Accuracy | {val_b_mean:.2f}% | {val_b_std:.2f} |\n")
            else:
                f.write(f"| Final Task B Accuracy | {val_b_mean} | {val_b_std} |\n")
                
            f.write(f"| Combined Accuracy | {avg.get('eval_all_mean', 'N/A'):.2f}% | {avg.get('eval_all_std', 'N/A'):.2f} |\n")
            f.write("\n")
            
        # Freezing Section
        if freezing_results:
            f.write("## Freezing Experiments Summary\n\n")
            f.write("| Percentile | Task A Retention (Mean ± Std) | Task B Accuracy (Mean ± Std) | Combined (Mean ± Std) |\n")
            f.write("| :--- | :--- | :--- | :--- |\n")
            
            for p, data in freezing_results:
                if "average" in data:
                    avg = data["average"]
                    a_mean = avg.get('final_task_a_mean', 0)
                    a_std = avg.get('final_task_a_std', 0)
                    
                    b_mean_list = avg.get('task_b_mean', [])
                    b_std_list = avg.get('task_b_std', [])
                    b_mean = b_mean_list[-1] if isinstance(b_mean_list, list) and b_mean_list else 0
                    b_std = b_std_list[-1] if isinstance(b_std_list, list) and b_std_list else 0
                    
                    c_mean = avg.get('eval_all_mean', 0)
                    c_std = avg.get('eval_all_std', 0)
                    
                    f.write(f"| {p:.2f} | {a_mean:.2f} ± {a_std:.2f} | {b_mean:.2f} ± {b_std:.2f} | {c_mean:.2f} ± {c_std:.2f} |\n")
            f.write("\n")

        # Random Freezing Section
        if random_results:
            f.write("## Random Freezing Experiments Summary\n\n")
            f.write("| Percentile | Task A Retention (Mean ± Std) | Task B Accuracy (Mean ± Std) | Combined (Mean ± Std) |\n")
            f.write("| :--- | :--- | :--- | :--- |\n")
            
            for p, data in random_results:
                if "average" in data:
                    avg = data["average"]
                    a_mean = avg.get('final_task_a_mean', 0)
                    a_std = avg.get('final_task_a_std', 0)
                    
                    b_mean_list = avg.get('task_b_mean', [])
                    b_std_list = avg.get('task_b_std', [])
                    b_mean = b_mean_list[-1] if isinstance(b_mean_list, list) and b_mean_list else 0
                    b_std = b_std_list[-1] if isinstance(b_std_list, list) and b_std_list else 0
                    
                    c_mean = avg.get('eval_all_mean', 0)
                    c_std = avg.get('eval_all_std', 0)
                    
                    f.write(f"| {p:.2f} | {a_mean:.2f} ± {a_std:.2f} | {b_mean:.2f} ± {b_std:.2f} | {c_mean:.2f} ± {c_std:.2f} |\n")
            f.write("\n")

        # Index Freezing Section
        if index_results:
            f.write("## Index Freezing Experiments Summary\n\n")
            f.write("| Percentile | Task A Retention (Mean ± Std) | Task B Accuracy (Mean ± Std) | Combined (Mean ± Std) |\n")
            f.write("| :--- | :--- | :--- | :--- |\n")
            
            for p, data in index_results:
                if "average" in data:
                    avg = data["average"]
                    a_mean = avg.get('final_task_a_mean', 0)
                    a_std = avg.get('final_task_a_std', 0)
                    
                    b_mean_list = avg.get('task_b_mean', [])
                    b_std_list = avg.get('task_b_std', [])
                    b_mean = b_mean_list[-1] if isinstance(b_mean_list, list) and b_mean_list else 0
                    b_std = b_std_list[-1] if isinstance(b_std_list, list) and b_std_list else 0
                    
                    c_mean = avg.get('eval_all_mean', 0)
                    c_std = avg.get('eval_all_std', 0)
                    
                    f.write(f"| {p:.2f} | {a_mean:.2f} ± {a_std:.2f} | {b_mean:.2f} ± {b_std:.2f} | {c_mean:.2f} ± {c_std:.2f} |\n")
            f.write("\n")

        # NoScale Freezing Section
        if noscale_results:
            f.write("## NoScale Freezing Experiments Summary\n\n")
            f.write("| Percentile | Task A Retention (Mean ± Std) | Task B Accuracy (Mean ± Std) | Combined (Mean ± Std) |\n")
            f.write("| :--- | :--- | :--- | :--- |\n")
            
            for p, data in noscale_results:
                if "average" in data:
                    avg = data["average"]
                    a_mean = avg.get('final_task_a_mean', 0)
                    a_std = avg.get('final_task_a_std', 0)
                    
                    b_mean_list = avg.get('task_b_mean', [])
                    b_std_list = avg.get('task_b_std', [])
                    b_mean = b_mean_list[-1] if isinstance(b_mean_list, list) and b_mean_list else 0
                    b_std = b_std_list[-1] if isinstance(b_std_list, list) and b_std_list else 0
                    
                    c_mean = avg.get('eval_all_mean', 0)
                    c_std = avg.get('eval_all_std', 0)
                    
                    f.write(f"| {p:.2f} | {a_mean:.2f} ± {a_std:.2f} | {b_mean:.2f} ± {b_std:.2f} | {c_mean:.2f} ± {c_std:.2f} |\n")
            f.write("\n")

        # Best Retention Runs (Freezing)
        if freezing_results:
            f.write("## Best Retention Runs (Freezing)\n\n")
            f.write("| Percentile | Best Retention (Task A) | Corresponding Task B | Corresponding Combined |\n")
            f.write("| :--- | :--- | :--- | :--- |\n")

            for p, data in freezing_results:
                if "runs" in data:
                    runs = data["runs"]
                    if not runs:
                        continue
                    
                    best_run = max(runs, key=lambda x: x.get("final_task_a", -1))
                    best_a = best_run.get("final_task_a", 0)
                    
                    tb_list = best_run.get("task_b", [])
                    best_b = tb_list[-1] if isinstance(tb_list, list) and tb_list else 0
                    
                    best_c = best_run.get("eval_all", 0)
                    
                    f.write(f"| {p:.2f} | {best_a:.2f}% | {best_b:.2f}% | {best_c:.2f}% |\n")

    print(f"Summary saved to {readme_path}")

def generate_root_readme():
    """Aggregates results from all results/results_epochs_* directories into results/README.md"""
    root_readme_path = "results/README.md"
    
    # Find all results directories
    all_dirs = glob.glob("results/results_epochs_*")
    epoch_dirs = []
    for d in all_dirs:
        if os.path.isdir(d):
            match = re.search(r'results_epochs_(\d+)', d)
            if match:
                epochs = int(match.group(1))
                epoch_dirs.append((epochs, d))
    
    epoch_dirs.sort(key=lambda x: x[0])
    
    content = "# MNIST with SNN - Continual Learning Experiments Results\n\n"
    content += "Aggregated results for mitigating catastrophic forgetting in SNNs.\n\n"
    
    for epochs, results_dir in epoch_dirs:
        content += f"### Results for {epochs} Epochs per Task\n\n"
        
        # Baseline
        baseline_path = os.path.join(results_dir, "cl_baseline.json")
        if os.path.exists(baseline_path):
            try:
                baseline_data = load_json(baseline_path)
                if "average" in baseline_data:
                    avg = baseline_data["average"]
                    content += "**Baseline:**\n"
                    content += f"- Final Task A Retention: {avg.get('final_task_a_mean', 'N/A'):.2f}% ± {avg.get('final_task_a_std', 'N/A'):.2f}\n"
                    content += f"- Combined Accuracy: {avg.get('eval_all_mean', 'N/A'):.2f}%\n\n"
            except Exception as e:
                print(f"Error reading baseline for {results_dir}: {e}")

        # Freezing
        freezing_files = glob.glob(os.path.join(results_dir, "freezing_*.json"))
        freezing_results = []
        for f in freezing_files:
            try:
                filename = os.path.basename(f)
                percentile_str = filename.replace("freezing_", "").replace(".json", "")
                percentile = float(percentile_str) / 100.0
                data = load_json(f)
                freezing_results.append((percentile, data))
            except ValueError:
                continue
        
        freezing_results.sort(key=lambda x: x[0])
        
        if freezing_results:
            content += "**Freezing Experiments:**\n\n"
            content += "| Percentile | Task A Retention | Task B Accuracy | Combined |\n"
            content += "| :--- | :--- | :--- | :--- |\n"
            
            for p, data in freezing_results:
                if "average" in data:
                    avg = data["average"]
                    a_mean = avg.get('final_task_a_mean', 0)
                    b_mean_list = avg.get('task_b_mean', [])
                    b_mean = b_mean_list[-1] if isinstance(b_mean_list, list) and b_mean_list else 0
                    c_mean = avg.get('eval_all_mean', 0)
                    
                    content += f"| {p:.2f} | {a_mean:.2f}% | {b_mean:.2f}% | {c_mean:.2f}% |\n"
            content += "\n"
            
        # Random Freezing
        random_files = glob.glob(os.path.join(results_dir, "random_*.json"))
        random_results = []
        for f in random_files:
            try:
                filename = os.path.basename(f)
                percentile_str = filename.replace("random_", "").replace(".json", "")
                percentile = float(percentile_str) / 100.0
                data = load_json(f)
                random_results.append((percentile, data))
            except ValueError:
                continue
        
        random_results.sort(key=lambda x: x[0])
        
        if random_results:
            content += "**Random Freezing Experiments:**\n\n"
            content += "| Percentile | Task A Retention | Task B Accuracy | Combined |\n"
            content += "| :--- | :--- | :--- | :--- |\n"
            
            for p, data in random_results:
                if "average" in data:
                    avg = data["average"]
                    a_mean = avg.get('final_task_a_mean', 0)
                    b_mean_list = avg.get('task_b_mean', [])
                    b_mean = b_mean_list[-1] if isinstance(b_mean_list, list) and b_mean_list else 0
                    c_mean = avg.get('eval_all_mean', 0)
                    
                    content += f"| {p:.2f} | {a_mean:.2f}% | {b_mean:.2f}% | {c_mean:.2f}% |\n"
            content += "\n"
            
        # Index Freezing
        index_files = glob.glob(os.path.join(results_dir, "index_*.json"))
        index_results = []
        for f in index_files:
            try:
                filename = os.path.basename(f)
                percentile_str = filename.replace("index_", "").replace(".json", "")
                percentile = float(percentile_str) / 100.0
                data = load_json(f)
                index_results.append((percentile, data))
            except ValueError:
                continue
        
        index_results.sort(key=lambda x: x[0])
        
        if index_results:
            content += "**Index Freezing Experiments:**\n\n"
            content += "| Percentile | Task A Retention | Task B Accuracy | Combined |\n"
            content += "| :--- | :--- | :--- | :--- |\n"
            
            for p, data in index_results:
                if "average" in data:
                    avg = data["average"]
                    a_mean = avg.get('final_task_a_mean', 0)
                    b_mean_list = avg.get('task_b_mean', [])
                    b_mean = b_mean_list[-1] if isinstance(b_mean_list, list) and b_mean_list else 0
                    c_mean = avg.get('eval_all_mean', 0)
                    
                    content += f"| {p:.2f} | {a_mean:.2f}% | {b_mean:.2f}% | {c_mean:.2f}% |\n"
            content += "\n"

        # NoScale Freezing
        noscale_files = glob.glob(os.path.join(results_dir, "noscale_*.json"))
        noscale_results = []
        for f in noscale_files:
            try:
                filename = os.path.basename(f)
                percentile_str = filename.replace("noscale_", "").replace(".json", "")
                percentile = float(percentile_str) / 100.0
                data = load_json(f)
                noscale_results.append((percentile, data))
            except ValueError:
                continue
        
        noscale_results.sort(key=lambda x: x[0])
        
        if noscale_results:
            content += "**NoScale Freezing Experiments:**\n\n"
            content += "| Percentile | Task A Retention | Task B Accuracy | Combined |\n"
            content += "| :--- | :--- | :--- | :--- |\n"
            
            for p, data in noscale_results:
                if "average" in data:
                    avg = data["average"]
                    a_mean = avg.get('final_task_a_mean', 0)
                    b_mean_list = avg.get('task_b_mean', [])
                    b_mean = b_mean_list[-1] if isinstance(b_mean_list, list) and b_mean_list else 0
                    c_mean = avg.get('eval_all_mean', 0)
                    
                    content += f"| {p:.2f} | {a_mean:.2f}% | {b_mean:.2f}% | {c_mean:.2f}% |\n"
            content += "\n"
            
        content += "---\n\n"

    with open(root_readme_path, "w") as f:
        f.write(content)
    
    print(f"Root README aggregated to {root_readme_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=str, nargs='?', help="Directory containing result JSON files")
    args = parser.parse_args()
    
    if args.results_dir:
        if os.path.exists(args.results_dir):
            generate_summary(args.results_dir)
        else:
            print(f"Directory not found: {args.results_dir}")
    else:
        # If no arg provided, try to generate root readme
        generate_root_readme()
