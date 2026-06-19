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
            if len(values[0]) > 0 and isinstance(values[0][0], list):
                # We handle 2D lists separately, so return empty for this path
                return [], []
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

    avg_history = {
        "eval_all_mean": np.mean([h["eval_all"] for h in histories]) if histories else 0.0,
        "eval_all_std": np.std([h["eval_all"] for h in histories]) if histories else 0.0,
        "final_task_a_mean": np.mean([h["final_task_a"] for h in histories]) if histories else 0.0,
        "final_task_a_std": np.std([h["final_task_a"] for h in histories]) if histories else 0.0,
    }

                        
    if any("full_curve" in h for h in histories):
        full_curve_mean, full_curve_std = safe_mean_std("full_curve")
        avg_history["full_curve_mean"] = full_curve_mean
        avg_history["full_curve_std"] = full_curve_std

    if any("task_b" in h for h in histories):
        task_b_mean, task_b_std = safe_mean_std("task_b")
        avg_history["task_b_mean"] = task_b_mean
        avg_history["task_b_std"] = task_b_std

    if any("full_curve_task_il" in h for h in histories):
        til_fc_mean, til_fc_std = safe_mean_std("full_curve_task_il")
        avg_history["full_curve_task_il_mean"] = til_fc_mean
        avg_history["full_curve_task_il_std"] = til_fc_std
    
    if any("task_b_task_il" in h for h in histories):
        til_tb_mean, til_tb_std = safe_mean_std("task_b_task_il")
        avg_history["task_b_task_il_mean"] = til_tb_mean
        avg_history["task_b_task_il_std"] = til_tb_std

    def safe_matrix_mean_std(key):
        values = [h[key] for h in histories if key in h]
        if not values or not values[0]:
            return [], []
        
        num_tasks = len(values[0])
        mean_matrix = []
        std_matrix = []
        for i in range(num_tasks):
            row_len = len(values[0][i])
            mean_row = []
            std_row = []
            for j in range(row_len):
                cell_vals = []
                for v in values:
                    if i < len(v) and j < len(v[i]):
                        cell_vals.append(v[i][j])
                if cell_vals:
                    mean_row.append(float(np.mean(cell_vals)))
                    std_row.append(float(np.std(cell_vals)))
                else:
                    mean_row.append(0.0)
                    std_row.append(0.0)
            mean_matrix.append(mean_row)
            std_matrix.append(std_row)
        return mean_matrix, std_matrix

    if any("acc_matrix_class_il" in h for h in histories):
        c_mean, c_std = safe_matrix_mean_std("acc_matrix_class_il")
        avg_history["acc_matrix_class_il_mean"] = c_mean
        avg_history["acc_matrix_class_il_std"] = c_std

    if any("acc_matrix_task_il" in h for h in histories):
        t_mean, t_std = safe_matrix_mean_std("acc_matrix_task_il")
        avg_history["acc_matrix_task_il_mean"] = t_mean
        avg_history["acc_matrix_task_il_std"] = t_std

    output_data = {
        "average": avg_history,
        "runs": histories
    }

    with open(filepath, 'w') as f:
        json.dump(output_data, f, indent=4)

