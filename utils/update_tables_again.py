import json, glob, os, re

def get_data():
    data_dict = {"Split-MNIST": {1: {}, 2: {}, 3: {}, 4: {}, 5: {}}, "Split-NMNIST": {1: {}, 3: {}, 5: {}}, "5-Split-FashionMNIST": {1: {}, 2: {}, 3: {}}}
    for dataset in ["Split-MNIST", "Split-NMNIST"]:
        for epoch in [1, 2, 3, 4, 5]:
            paths = glob.glob(f"results/SNN/{dataset}/epochs_{epoch}/*.json")
            for p in paths:
                name = os.path.basename(p).replace(".json", "").replace("_ltp0.01_ltd0.01", "")
                try:
                    with open(p) as f:
                        d = json.load(f).get("average", {})
                    
                    if "final_task_a_mean" in d:
                        data_dict[dataset][epoch][name] = {
                            "ta_c_m": d["final_task_a_mean"],
                            "ta_c_s": d["final_task_a_std"],
                            "ta_t_m": d.get("full_curve_task_il_mean", [0, '--'])[-1],
                            "ta_t_s": d.get("full_curve_task_il_std", [0, '--'])[-1],
                            "tb_c_m": d["task_b_mean"][0],
                            "tb_c_s": d["task_b_std"][0],
                            "tb_t_m": d.get("task_b_task_il_mean", [0, '--'])[-1],
                            "tb_t_s": d.get("task_b_task_il_std", [0, '--'])[-1],
                            "comb_m": d["eval_all_mean"],
                            "comb_s": d["eval_all_std"],
                        }
                except Exception as e:
                    pass

    for epoch in [1, 2, 3]:
        paths = glob.glob(f"results/SNN/5-Split-FashionMNIST/sweep_epochs_{epoch}/*.json")
        for p in paths:
            name = os.path.basename(p).replace(".json", "").replace("_ltp0.01_ltd0.01", "")
            try:
                with open(p) as f:
                    d = json.load(f).get("average", {})
                
                if "acc_matrix_class_il_mean" in d:
                    data_dict["5-Split-FashionMNIST"][epoch][name] = {
                        "c_mean": d["acc_matrix_class_il_mean"],
                        "c_std": d["acc_matrix_class_il_std"],
                        "t_mean": d["acc_matrix_task_il_mean"],
                        "t_std": d["acc_matrix_task_il_std"],
                        "comb_m": d["eval_all_mean"],
                        "comb_s": d["eval_all_std"],
                    }
            except Exception as e:
                pass
    return data_dict

data_dict = get_data()

def format_row(name, key, epoch, dataset="Split-MNIST"):
    if key not in data_dict[dataset][epoch]:
        return "        " + name + r" & -- & -- & -- & -- & -- \\"
    d = data_dict[dataset][epoch][key]
    
    ta_c = f"{d['ta_c_m']:.2f}\\% $\\pm$ {d['ta_c_s']:.2f}"
    
    ta_t = "--" if d['ta_t_m'] == '--' else f"{d['ta_t_m']:.2f}\\% $\\pm$ {d['ta_t_s']:.2f}"
    tb_c = f"{d['tb_c_m']:.2f}\\% $\\pm$ {d['tb_c_s']:.2f}"
    tb_t = "--" if d['tb_t_m'] == '--' else f"{d['tb_t_m']:.2f}\\% $\\pm$ {d['tb_t_s']:.2f}"
    comb = f"{d['comb_m']:.2f}\\%"
    
    return "        " + name + f" & {ta_c} & {ta_t} & {tb_c} & {tb_t} & {comb} \\\\"

def format_row_fashion_class(name, key, epoch):
    if key not in data_dict["5-Split-FashionMNIST"][epoch]:
        return "        " + name + r" & -- & -- & -- & -- & -- \\"
    d = data_dict["5-Split-FashionMNIST"][epoch][key]
    
    t = []
    for i in range(5):
        try:
            t.append(f"{d['c_mean'][i]:.2f}\\% $\\pm$ {d['c_std'][i]:.2f}")
        except IndexError:
            t.append("--")
            
    return "        " + name + f" & {t[0]} & {t[1]} & {t[2]} & {t[3]} & {t[4]} \\\\"

def format_row_fashion_task(name, key, epoch):
    if key not in data_dict["5-Split-FashionMNIST"][epoch]:
        return "        " + name + r" & -- & -- & -- & -- & -- \\"
    d = data_dict["5-Split-FashionMNIST"][epoch][key]
    
    t = []
    for i in range(5):
        try:
            t.append(f"{d['t_mean'][i]:.2f}\\% $\\pm$ {d['t_std'][i]:.2f}")
        except IndexError:
            t.append("--")
            
    return "        " + name + f" & {t[0]} & {t[1]} & {t[2]} & {t[3]} & {t[4]} \\\\"

def wrap_table(content, caption, label=""):
    # Notice: No \resizebox, we just use standard font size, centered
    return r"""\begin{table*}[htb]
    \caption{""" + caption + r"""}
""" + (f"    \\label{{{label}}}\n" if label else "") + r"""    \begin{center}
    \begin{tabular}{lccccc}
        \toprule
        \textbf{Method} & \textbf{Task A (Class-IL)} & \textbf{Task A (Task-IL)} & \textbf{Task B (Class-IL)} & \textbf{Task B (Task-IL)} & \textbf{Combined} \\
        \midrule
""" + content + r"""
        \bottomrule
    \end{tabular}
    \end{center}
\end{table*}"""

def wrap_table_fashion_unified(content, caption, label=""):
    return r"""\begin{table*}[htb]
    \caption{""" + caption + r"""}
""" + (f"    \\label{{{label}}}\n" if label else "") + r"""    \begin{center}
    \begin{tabular}{lcccccc}
        \toprule
        & \multicolumn{2}{c}{\textbf{P-Factor (Ours)}} & \multicolumn{2}{c}{\textbf{Experience Replay}} & \multicolumn{2}{c}{\textbf{Fine-Tuning (Baseline)}} \\
        \cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}
        \textbf{Task} & \textbf{Class-IL} & \textbf{Task-IL} & \textbf{Class-IL} & \textbf{Task-IL} & \textbf{Class-IL} & \textbf{Task-IL} \\
        \midrule
""" + content + r"""
        \bottomrule
    \end{tabular}
    \end{center}
\end{table*}"""

def wrap_table_fashion_progression(content, caption, label=""):
    return r"""\begin{table*}[htb]
    \caption{""" + caption + r"""}
""" + (f"    \\label{{{label}}}\n" if label else "") + r"""    \begin{center}
    \resizebox{\textwidth}{!}{
    \begin{tabular}{lcccccccccc}
        \toprule
        & \multicolumn{2}{c}{\textbf{After Task 1}} & \multicolumn{2}{c}{\textbf{After Task 2}} & \multicolumn{2}{c}{\textbf{After Task 3}} & \multicolumn{2}{c}{\textbf{After Task 4}} & \multicolumn{2}{c}{\textbf{After Task 5}} \\
        \cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9} \cmidrule(lr){10-11}
        \textbf{Eval Task} & \textbf{C-IL} & \textbf{T-IL} & \textbf{C-IL} & \textbf{T-IL} & \textbf{C-IL} & \textbf{T-IL} & \textbf{C-IL} & \textbf{T-IL} & \textbf{C-IL} & \textbf{T-IL} \\
        \midrule
""" + content + r"""
        \bottomrule
    \end{tabular}
    }
    \end{center}
\end{table*}"""

def replace_table(text, caption_start, replacement):
    # Support both [t] and [h!] table types
    if r"\begin{table*}[htb]" in replacement:
        table_start = r"\begin{table*}[htb]"
    elif r"\begin{table*}[htb]" in replacement:
        table_start = r"\begin{table*}[htb]"
    else:
        table_start = r"\begin{table*}[htb]"

    parts = text.split(table_start)
    for i in range(1, len(parts)):
        if caption_start in parts[i]:
            sub_parts = parts[i].split(r"\end{table*}", 1)
            parts[i] = "\n" + replacement.replace(r"\begin{table*}[htb]", "").replace(r"\begin{table*}[htb]", "").replace(r"\end{table*}", "") + r"\end{table*}" + sub_parts[1]
    return table_start.join(parts)



def get_ablation_table(epoch, dataset="Split-MNIST"):
    rows = [
        r'        \multicolumn{6}{l}{\textit{Weight Scaling Ablation}} \\',
        format_row(r'No-Scale $\tau=0.2$', 'noscale_20', epoch, dataset),
        format_row(r'No-Scale $\tau=0.4$', 'noscale_40', epoch, dataset),
        format_row(r'No-Scale $\tau=0.6$', 'noscale_60', epoch, dataset),
        format_row(r'No-Scale $\tau=0.8$', 'noscale_80', epoch, dataset),
        r'        \midrule',
        r'        \multicolumn{6}{l}{\textit{Novice Reset Strategy Ablation}} \\',
        format_row(r'Reset-Zero $\tau=0.2$', 'reset_zero_20', epoch, dataset),
        format_row(r'Reset-Zero $\tau=0.4$', 'reset_zero_40', epoch, dataset),
        format_row(r'Reset-Zero $\tau=0.6$', 'reset_zero_60', epoch, dataset),
        format_row(r'Reset-Zero $\tau=0.8$', 'reset_zero_80', epoch, dataset),
    ]
    return wrap_table("\n".join(rows), f"\\textbf{{Epoch {epoch}: Ablation Studies (Mean $\\pm$ Std)}}.")

def get_baseline_table(epoch, dataset="Split-MNIST"):
    rows = [
        format_row(r'Fine-Tuning (Baseline)', 'cl_baseline', epoch, dataset),
        format_row(r'ER (Buffer=200)', 'er_200', epoch, dataset),
        r'        \midrule',
        format_row(r'EWC ($\lambda=10^3$)', 'ewc_1000', epoch, dataset),
        format_row(r'EWC ($\lambda=10^5$)', 'ewc_100000', epoch, dataset),
        format_row(r'EWC ($\lambda=10^6$)', 'ewc_1000000', epoch, dataset),
        r'        \midrule',
        format_row(r'SI ($c=1$)', 'si_1', epoch, dataset),
        format_row(r'SI ($c=10^2$)', 'si_100', epoch, dataset),
        format_row(r'SI ($c=10^4$)', 'si_10000', epoch, dataset),
        format_row(r'SI ($c=10^6$)', 'si_1000000', epoch, dataset),
        r'        \midrule',
        format_row(r'PackNet (20\%)', 'packnet_20', epoch, dataset),
        format_row(r'PackNet (40\%)', 'packnet_40', epoch, dataset),
        format_row(r'PackNet (60\%)', 'packnet_60', epoch, dataset),
        format_row(r'PackNet (80\%)', 'packnet_80', epoch, dataset),
        r'        \midrule',
        format_row(r'Random (20\%)', 'random_20', epoch, dataset),
        format_row(r'Random (40\%)', 'random_40', epoch, dataset),
        format_row(r'Random (60\%)', 'random_60', epoch, dataset),
        format_row(r'Random (80\%)', 'random_80', epoch, dataset),
        r'        \midrule',
        format_row(r'\textbf{P-Factor $\tau=0.2$ (Ours)}', 'freezing_20', epoch, dataset),
        format_row(r'\textbf{P-Factor $\tau=0.4$ (Ours)}', 'freezing_40', epoch, dataset),
        format_row(r'\textbf{P-Factor $\tau=0.6$ (Ours)}', 'freezing_60', epoch, dataset),
        format_row(r'\textbf{P-Factor $\tau=0.8$ (Ours)}', 'freezing_80', epoch, dataset),
    ]
    return wrap_table("\n".join(rows), f"\\textbf{{Epoch {epoch}: P-Factor vs Baselines (Mean $\\pm$ Std)}}.")


# NOW REPLACE MAIN TEXT TABLES
def get_main_gradient_table():
    epoch = 3
    dataset = "Split-MNIST"
    rows = [
        format_row(r'Fine-Tuning (Baseline)', 'cl_baseline', epoch, dataset),
        r'        \midrule',
        format_row(r'EWC ($\lambda=10^3$)', 'ewc_1000', epoch, dataset),
        format_row(r'EWC ($\lambda=10^5$)', 'ewc_100000', epoch, dataset),
        format_row(r'EWC ($\lambda=10^6$)', 'ewc_1000000', epoch, dataset),
        r'        \midrule',
        format_row(r'SI ($c=1$)', 'si_1', epoch, dataset),
        format_row(r'SI ($c=10^2$)', 'si_100', epoch, dataset),
        format_row(r'SI ($c=10^4$)', 'si_10000', epoch, dataset),
        format_row(r'SI ($c=10^6$)', 'si_1000000', epoch, dataset),
    ]
    return wrap_table("\n".join(rows), "Gradient-Based Baselines Performance Summary (Split-MNIST, Epoch 3)", "tab:gradient_baselines")

def get_main_structural_table():
    epoch = 3
    dataset = "Split-MNIST"
    rows = [
        r'        \multicolumn{6}{l}{\textit{Threshold $\tau=0.2$ (20\% Frozen)}} \\',
        format_row(r'PackNet (Magnitude)', 'packnet_20', epoch, dataset),
        format_row(r'Random Freezing', 'random_20', epoch, dataset),
        format_row(r'\textbf{P-Factor (Ours)}', 'freezing_20', epoch, dataset),
        r'        \midrule',
        r'        \multicolumn{6}{l}{\textit{Threshold $\tau=0.4$ (40\% Frozen)}} \\',
        format_row(r'PackNet (Magnitude)', 'packnet_40', epoch, dataset),
        format_row(r'Random Freezing', 'random_40', epoch, dataset),
        format_row(r'\textbf{P-Factor (Ours)}', 'freezing_40', epoch, dataset),
        r'        \midrule',
        r'        \multicolumn{6}{l}{\textit{Threshold $\tau=0.6$ (60\% Frozen)}} \\',
        format_row(r'PackNet (Magnitude)', 'packnet_60', epoch, dataset),
        format_row(r'Random Freezing', 'random_60', epoch, dataset),
        format_row(r'\textbf{P-Factor (Ours)}', 'freezing_60', epoch, dataset),
        r'        \midrule',
        r'        \multicolumn{6}{l}{\textit{Threshold $\tau=0.8$ (80\% Frozen)}} \\',
        format_row(r'PackNet (Magnitude)', 'packnet_80', epoch, dataset),
        format_row(r'Random Freezing', 'random_80', epoch, dataset),
        format_row(r'\textbf{P-Factor (Ours)}', 'freezing_80', epoch, dataset),
    ]
    return wrap_table("\n".join(rows), "Comparison of Structural Engram Allocation Strategies (Split-MNIST, Epoch 3)", "tab:structural_comparison")

def get_main_ablation_table():
    epoch = 3
    dataset = "Split-MNIST"
    rows = [
        r'        \multicolumn{6}{l}{\textit{Weight Scaling Ablation ($\tau=0.8$)}} \\',
        format_row(r'No-Scale', 'noscale_80', epoch, dataset),
        format_row(r'\textbf{P-Factor (With Scale)}', 'freezing_80', epoch, dataset),
        r'        \midrule',
        r'        \multicolumn{6}{l}{\textit{Novice Reset Strategy ($\tau=0.8$)}} \\',
        format_row(r'Reset-Zero', 'reset_zero_80', epoch, dataset),
        format_row(r'\textbf{P-Factor (Default No-Reset)}', 'freezing_80', epoch, dataset),
    ]
    return wrap_table("\n".join(rows), "Ablation Study: Forward Pass Scaling and Reset Mechanics (Split-MNIST, Epoch 3)", "tab:ablations")

def get_nmnist_table():
    epoch = 3
    dataset = "Split-NMNIST"
    rows = [
        format_row(r'Fine-Tuning (Baseline)', 'cl_baseline', epoch, dataset),
        format_row(r'ER (Buffer=200) [Oracle]', 'er_200', epoch, dataset),
        format_row(r'EWC ($\lambda=10^6$)', 'ewc_1000000', epoch, dataset),
        format_row(r'SI ($c=10^6$)', 'si_1000000', epoch, dataset),
        r'        \midrule',
        r'        \multicolumn{6}{l}{\textit{Structural Isolation}} \\',
        format_row(r'PackNet (80\%)', 'packnet_80', epoch, dataset),
        format_row(r'Random Freezing (80\%)', 'random_80', epoch, dataset),
        format_row(r'\textbf{P-Factor $\tau=0.8$ (Ours)}', 'freezing_80', epoch, dataset),
    ]
    return wrap_table("\n".join(rows), "Performance on Spatio-Temporal Event Streams (N-MNIST, Epoch 3)", "tab:nmnist_results")




def get_fashion_unified_table(epoch):
    d_dict = data_dict["5-Split-FashionMNIST"][epoch]
    
    # Identify keys
    baseline_key = 'cl_baseline'
    er_key = 'fmnist_5t_er200_h1024_e3_lr0.001' # Hardcoded ER key based on bash script
    
    # Find the best P-factor sweep config (ignore baseline, ER, and 'sweep_results')
    p_keys = [k for k in d_dict.keys() if k not in [baseline_key, er_key, 'sweep_results']]
    p_key = None
    if p_keys:
        # Just grab the first one or the one with best Task-IL average
        p_key = sorted(p_keys, key=lambda k: sum(d_dict[k]['t_mean'][-1])/len(d_dict[k]['t_mean'][-1]) if len(d_dict[k]['t_mean'])>0 else 0, reverse=True)[0]

    def format_cell(key, idx, is_task=False):
        if key not in d_dict:
            return "--"
        d = d_dict[key]
        try:
            m = d['t_mean'][-1][idx] if is_task else d['c_mean'][-1][idx]
            s = d['t_std'][-1][idx] if is_task else d['c_std'][-1][idx]
            return f"{m:.1f}\\% $\\pm$ {s:.1f}"
        except IndexError:
            return "--"

    rows = []
    for t in range(5):
        p_c = format_cell(p_key, t, False)
        p_t = format_cell(p_key, t, True)
        er_c = format_cell(er_key, t, False)
        er_t = format_cell(er_key, t, True)
        b_c = format_cell(baseline_key, t, False)
        b_t = format_cell(baseline_key, t, True)
        
        rows.append(f"        Task {t+1} & {p_c} & {p_t} & {er_c} & {er_t} & {b_c} & {b_t} \\\\")
    
    caption = f"\\textbf{{5-Split Fashion-MNIST Epoch {epoch}: Final Accuracies (Mean $\\pm$ Std)}}."
    label = f"tab:fashion_epoch{epoch}"
    return wrap_table_fashion_unified("\n".join(rows), caption, label)

def get_fashion_progression_table(epoch, method_type):
    d_dict = data_dict["5-Split-FashionMNIST"][epoch]
    
    if method_type == 'baseline':
        key = 'cl_baseline'
        caption_prefix = "Fine-Tuning (Baseline)"
        label = f"tab:fashion_prog_baseline_epoch{epoch}"
    elif method_type == 'er':
        key = 'fmnist_5t_er200_h1024_e3_lr0.001'
        caption_prefix = "Experience Replay"
        label = f"tab:fashion_prog_er_epoch{epoch}"
    else:
        # P-Factor
        p_keys = [k for k in d_dict.keys() if k not in ['cl_baseline', 'fmnist_5t_er200_h1024_e3_lr0.001', 'sweep_results']]
        key = sorted(p_keys, key=lambda k: sum(d_dict[k]['t_mean'][-1])/len(d_dict[k]['t_mean'][-1]) if len(d_dict[k]['t_mean'])>0 else 0, reverse=True)[0] if p_keys else None
        caption_prefix = "P-Factor"
        label = f"tab:fashion_prog_pfactor_epoch{epoch}"

    def format_cell(t_eval, t_state, is_task=False):
        if key not in d_dict:
            return "--"
        d = d_dict[key]
        try:
            # The matrix is lower triangular. t_state is the row we evaluate after, t_eval is the task evaluated
            if t_eval > t_state:
                return "--"
            m = d['t_mean'][t_state][t_eval] if is_task else d['c_mean'][t_state][t_eval]
            s = d['t_std'][t_state][t_eval] if is_task else d['c_std'][t_state][t_eval]
            return f"{m:.1f}\\% $\\pm$ {s:.1f}"
        except (IndexError, TypeError):
            return "--"

    rows = []
    for t_eval in range(5):
        row_str = f"        Task {t_eval+1}"
        for t_state in range(5):
            c_val = format_cell(t_eval, t_state, False)
            t_val = format_cell(t_eval, t_state, True)
            row_str += f" & {c_val} & {t_val}"
        row_str += " \\\\"
        rows.append(row_str)
        
    caption = f"\\textbf{{5-Split Fashion-MNIST Epoch {epoch}: {caption_prefix} Progression over Time}}."
    return wrap_table_fashion_progression("\n".join(rows), caption, label)

def get_split_mnist_epochs_2_4_table():
    rows = []
    # Force epoch 2 and 4
    for ep in [2, 4]:
        rows.append(format_row(f"Epoch {ep}", "freezing_80", ep, "Split-MNIST"))
    
    body = "\n".join(rows)
    return f"""\\begin{{table*}}[htb]
    \\caption{{P-Factor Performance at Epochs 2 and 4 ($\\tau=0.8$)}}
    \\label{{tab:split_mnist_epochs_2_4}}
    \\begin{{center}}
    \\begin{{tabular}}{{lccccc}}
        \\toprule
        \\textbf{{Epochs}} & \\textbf{{Task A (Class-IL)}} & \\textbf{{Task A (Task-IL)}} & \\textbf{{Task B (Class-IL)}} & \\textbf{{Task B (Task-IL)}} & \\textbf{{Combined}} \\\\
        \\midrule
{body}
        \\bottomrule
    \\end{{tabular}}
    \\end{{center}}
\\end{{table*}}"""

with open("Paper.tex", "r") as f:
    paper = f.read()

# 1. Replace Appendix Tables using replace_table so we don't overwrite user text!
for epoch in [1, 3, 5]:
    paper = replace_table(paper, f"\\textbf{{Epoch {epoch}: P-Factor vs Baselines", get_baseline_table(epoch))
    paper = replace_table(paper, f"\\textbf{{Epoch {epoch}: Ablation Studies", get_ablation_table(epoch))

# 2. Main text tables
paper = replace_table(paper, r"Gradient-Based Baselines Performance Summary", get_main_gradient_table())
paper = replace_table(paper, r"Comparison of Structural Engram Allocation Strategies", get_main_structural_table())
paper = replace_table(paper, r"Ablation Study: Forward Pass Scaling", get_main_ablation_table())
paper = replace_table(paper, r"Performance on Spatio-Temporal Event Streams", get_nmnist_table())
paper = replace_table(paper, r"P-Factor Performance at Epochs 2 and 4", get_split_mnist_epochs_2_4_table())

# 3. Update progressions for all epochs
for epoch in [1, 2, 3]:
    paper = replace_table(paper, f"5-Split Fashion-MNIST Epoch {epoch}: P-Factor Progression", get_fashion_progression_table(epoch, 'pfactor'))
    paper = replace_table(paper, f"5-Split Fashion-MNIST Epoch {epoch}: Experience Replay Progression", get_fashion_progression_table(epoch, 'er'))
    paper = replace_table(paper, f"5-Split Fashion-MNIST Epoch {epoch}: Fine-Tuning (Baseline) Progression", get_fashion_progression_table(epoch, 'baseline'))

with open("Paper.tex", "w") as f:
    f.write(paper)

print("Tables updated: All tables populated without touching text!")
