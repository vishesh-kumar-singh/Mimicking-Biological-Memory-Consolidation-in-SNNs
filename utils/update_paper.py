import json
import os
import re

def parse_json(filepath):
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    avg = data.get("average", {})
    if not avg:
        return None
        
    task_a_cil_m = avg.get("final_task_a_mean", 0)
    task_a_cil_s = avg.get("final_task_a_std", 0)
    
    task_a_til_m = avg.get("full_curve_task_il_mean", [0])[-1]
    task_a_til_s = avg.get("full_curve_task_il_std", [0])[-1] if "full_curve_task_il_std" in avg else 0
    
    task_b_cil_m = avg.get("task_b_mean", [0])[-1]
    task_b_cil_s = avg.get("task_b_std", [0])[-1]
    
    task_b_til_m = avg.get("task_b_task_il_mean", [0])[-1] if "task_b_task_il_mean" in avg else 0
    task_b_til_s = avg.get("task_b_task_il_std", [0])[-1] if "task_b_task_il_std" in avg else 0
    
    comb_m = avg.get("eval_all_mean", 0)
    
    return f"{task_a_cil_m:.2f}\\% $\\pm$ {task_a_cil_s:.2f} & {task_a_til_m:.2f}\\% $\\pm$ {task_a_til_s:.2f} & {task_b_cil_m:.2f}\\% $\\pm$ {task_b_cil_s:.2f} & {task_b_til_m:.2f}\\% $\\pm$ {task_b_til_s:.2f} & {comb_m:.2f}\\% \\\\"

def update_paper():
    with open("Paper.tex", "r") as f:
        content = f.read()

    base_dir = "results/SNN/Split-MNIST/epochs_1"
    
    mapping = {
        "0.2": os.path.join(base_dir, "freezing_20_ltp0.01_ltd0.01.json"),
        "0.4": os.path.join(base_dir, "freezing_40_ltp0.01_ltd0.01.json"),
        "0.6": os.path.join(base_dir, "freezing_60_ltp0.01_ltd0.01.json"),
        "0.8": os.path.join(base_dir, "freezing_80_ltp0.01_ltd0.01.json"),
    }
    
    # Isolate Epoch 1 Section to avoid overwriting other tables (like N-MNIST)
    pattern_section = re.compile(r"(\\subsubsection\{Epoch 1 Results\}.*?\\end\{table\*\})", re.DOTALL)
    match = pattern_section.search(content)
    if not match:
        print("Could not find Epoch 1 Results block")
        return

    section_content = match.group(1)
    
    for tau, json_path in mapping.items():
        row_content = parse_json(json_path)
        if row_content:
            pattern = re.compile(r"\\textbf\{P-Factor \$\\tau=" + re.escape(tau) + r"\$ \(Ours\)\} & .*?\\\\")
            replacement = f"\\textbf{{P-Factor $\\tau={tau}$ (Ours)}} & {row_content}"
            section_content = pattern.sub(lambda m: replacement, section_content)

    new_content = content[:match.start()] + section_content + content[match.end():]

    with open("Paper.tex", "w") as f:
        f.write(new_content)
    print("Updated Paper.tex with latest JSON data.")

if __name__ == "__main__":
    update_paper()
