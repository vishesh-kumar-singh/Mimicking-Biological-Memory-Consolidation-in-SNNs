import os
import json
import numpy as np
import glob
import re

def load_legacy_json(directory, pattern):
    """Loads legacy JSON result files matching a pattern."""
    files = glob.glob(os.path.join(directory, pattern))
    histories = []
    for f in files:
        try:
            with open(f, 'r') as fp:
                histories.append(json.load(fp))
        except Exception as e:
            print(f"Error loading {f}: {e}")
    return histories

def parse_results_file(filepath):
    """Parses the JSON result file to extract individual run histories."""
    if not os.path.exists(filepath):
        return []
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            return data.get("runs", [])
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return []

def save_aggregated_results(filepath, histories):
    """Calculates averages and writes the aggregated results file as JSON."""
    if not histories:
        return

    def safe_mean_std(key):
                                             
        values = [h[key] for h in histories if key in h]
        if not values:
            return [], []
        
                                   
        if isinstance(values[0], list):
            max_len = max(len(v) for v in values)
                           
            padded = []
            for v in values:
                p = v + [np.nan] * (max_len - len(v))
                padded.append(p)
            
                                              
            mean = np.nanmean(padded, axis=0).tolist()
            std = np.nanstd(padded, axis=0).tolist()
            return mean, std
        else:
                           
            return np.mean(values), np.std(values)

                        
    full_curve_mean, full_curve_std = safe_mean_std("full_curve")
    task_b_mean, task_b_std = safe_mean_std("task_b")
    
    avg_history = {
        "full_curve_mean": full_curve_mean,
        "full_curve_std": full_curve_std,
        "task_b_mean": task_b_mean,
        "task_b_std": task_b_std,
        "eval_all_mean": np.mean([h["eval_all"] for h in histories]),
        "eval_all_std": np.std([h["eval_all"] for h in histories]),
        "final_task_a_mean": np.mean([h["final_task_a"] for h in histories]),
        "final_task_a_std": np.std([h["final_task_a"] for h in histories]),
    }

    output_data = {
        "average": avg_history,
        "runs": histories
    }

    with open(filepath, 'w') as f:
        json.dump(output_data, f, indent=4)
