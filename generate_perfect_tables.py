import json, glob, os, re

def get_data():
    data_dict = {"Split-MNIST": {1: {}, 3: {}, 5: {}}, "Split-NMNIST": {1: {}, 3: {}, 5: {}}}
    for dataset in ["Split-MNIST", "Split-NMNIST"]:
        for epoch in [1, 3, 5]:
            paths = glob.glob(f"results/SNN/{dataset}/epochs_{epoch}/*.json")
            for p in paths:
                name = os.path.basename(p).replace(".json", "")
                try:
                    with open(p) as f:
                        d = json.load(f).get("average", {})
                    
                    if "final_task_a_mean" in d:
                        data_dict[dataset][epoch][name] = {
                            "ta_c_m": d["final_task_a_mean"],
                            "ta_c_s": d["final_task_a_std"],
                            "ta_t_m": d.get("full_curve_task_il_mean", [0, '--'])[-1],
                            "ta_t_s": d.get("full_curve_task_il_std", [0, '--'])[-1],
                            "tb_m": d["task_b_mean"][0],
                            "tb_s": d["task_b_std"][0],
                            "comb_m": d["eval_all_mean"],
                            "comb_s": d["eval_all_std"],
                        }
                except Exception as e:
                    pass
    return data_dict

data_dict = get_data()

def format_row(name, key, epoch, dataset="Split-MNIST"):
    if key not in data_dict[dataset][epoch]:
        return "        " + name + r" & -- & -- & -- & -- \\"
    d = data_dict[dataset][epoch][key]
    
    ta_c = f"{d['ta_c_m']:.2f}\\% $\\pm$ {d['ta_c_s']:.2f}"
    
    if d['ta_t_m'] == '--':
        ta_t = "--"
    else:
        ta_t = f"{d['ta_t_m']:.2f}\\% $\\pm$ {d['ta_t_s']:.2f}"
        
    tb = f"{d['tb_m']:.2f}\\% $\\pm$ {d['tb_s']:.2f}"
    comb = f"{d['comb_m']:.2f}\\%"
    
    return "        " + name + f" & {ta_c} & {ta_t} & {tb} & {comb} \\\\"

def get_ablation_table(epoch, dataset="Split-MNIST"):
    rows = [
        r'        \multicolumn{5}{l}{\textit{Weight Scaling Ablation}} \\',
        format_row(r'No-Scale $\tau=0.2$', 'noscale_20', epoch, dataset),
        format_row(r'No-Scale $\tau=0.4$', 'noscale_40', epoch, dataset),
        format_row(r'No-Scale $\tau=0.6$', 'noscale_60', epoch, dataset),
        format_row(r'No-Scale $\tau=0.8$', 'noscale_80', epoch, dataset),
        r'        \midrule',
        r'        \multicolumn{5}{l}{\textit{Novice Reset Strategy}} \\',
        format_row(r'No-Reset $\tau=0.2$', 'noreset_20', epoch, dataset),
        format_row(r'No-Reset $\tau=0.4$', 'noreset_40', epoch, dataset),
        format_row(r'No-Reset $\tau=0.6$', 'noreset_60', epoch, dataset),
        format_row(r'No-Reset $\tau=0.8$', 'noreset_80', epoch, dataset),
        r'        \midrule',
        format_row(r'Reset-Zero $\tau=0.2$', 'reset_zero_20', epoch, dataset),
        format_row(r'Reset-Zero $\tau=0.4$', 'reset_zero_40', epoch, dataset),
        format_row(r'Reset-Zero $\tau=0.6$', 'reset_zero_60', epoch, dataset),
        format_row(r'Reset-Zero $\tau=0.8$', 'reset_zero_80', epoch, dataset),
        r'        \midrule',
        format_row(r'Reset-Scale $\tau=0.2$', 'reset_scale_20', epoch, dataset),
        format_row(r'Reset-Scale $\tau=0.4$', 'reset_scale_40', epoch, dataset),
        format_row(r'Reset-Scale $\tau=0.6$', 'reset_scale_60', epoch, dataset),
        format_row(r'Reset-Scale $\tau=0.8$', 'reset_scale_80', epoch, dataset),
    ]
    return r"""\begin{table}[H]
    \caption{\textbf{Epoch """ + str(epoch) + r""": Ablation Studies (Mean $\pm$ Std)}.}
    \begin{center}
    \resizebox{\columnwidth}{!}{
    \begin{tabular}{lcccc}
        \toprule
        \textbf{Method} & \textbf{Task A (Class-IL)} & \textbf{Task A (Task-IL)} & \textbf{Task B (Acc.)} & \textbf{Combined} \\
        \midrule
""" + "\n".join(rows) + r"""
        \bottomrule
    \end{tabular}}
    \end{center}
\end{table}"""

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
    return r"""\begin{table}[H]
    \caption{\textbf{Epoch """ + str(epoch) + r""": P-Factor vs Baselines (Mean $\pm$ Std)}.}
    \begin{center}
    \resizebox{\columnwidth}{!}{
    \begin{tabular}{lcccc}
        \toprule
        \textbf{Method} & \textbf{Task A (Class-IL)} & \textbf{Task A (Task-IL)} & \textbf{Task B (Acc.)} & \textbf{Combined} \\
        \midrule
""" + "\n".join(rows) + r"""
        \bottomrule
    \end{tabular}}
    \end{center}
\end{table}"""

def replace_section(text, start_marker, end_marker, replacement):
    parts = text.split(start_marker)
    if len(parts) < 2: return text
    pre = parts[0]
    post = parts[1].split(end_marker, 1)[1]
    return pre + start_marker + "\n" + replacement + "\n" + end_marker + post

with open("Paper.tex", "r") as f:
    paper = f.read()

# Replace Appendix Split-MNIST
rep1 = f"\\subsubsection{{Epoch 1 Results}}\n\n{get_baseline_table(1)}\n\n{get_ablation_table(1)}\n"
paper = replace_section(paper, "% ================= EPOCH 1 =================", "% ================= EPOCH 3 =================", rep1)

rep3 = f"\\subsubsection{{Epoch 3 Results}}\n\n{get_baseline_table(3)}\n\n{get_ablation_table(3)}\n"
paper = replace_section(paper, "% ================= EPOCH 3 =================", "% ================= EPOCH 5 =================", rep3)

rep5 = f"\\subsubsection{{Epoch 5 Results}}\n\n{get_baseline_table(5)}\n\n{get_ablation_table(5)}\n"
paper = replace_section(paper, "% ================= EPOCH 5 =================", "\\subsection{Experimental results for Split-N-MNIST}", rep5)

# NOW REPLACE MAIN TEXT TABLES using custom replacement to avoid regex issues
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
    return r"""\begin{table}[H]
    \caption{Gradient-Based Baselines Performance Summary (Split-MNIST, Epoch 3)}
    \begin{center}
    \resizebox{\columnwidth}{!}{
    \begin{tabular}{lcccc}
        \toprule
        \textbf{Method} & \textbf{Task A (Class-IL)} & \textbf{Task A (Task-IL)} & \textbf{Task B (Acc.)} & \textbf{Combined} \\
        \midrule
""" + "\n".join(rows) + r"""
        \bottomrule
    \end{tabular}}
    \label{tab:gradient_baselines}
    \end{center}
\end{table}"""

def get_main_structural_table():
    epoch = 3
    dataset = "Split-MNIST"
    rows = [
        r'        \multicolumn{5}{l}{\textit{Threshold $\tau=0.2$ (20\% Frozen)}} \\',
        format_row(r'PackNet (Magnitude)', 'packnet_20', epoch, dataset),
        format_row(r'Random Freezing', 'random_20', epoch, dataset),
        format_row(r'\textbf{P-Factor (Ours)}', 'freezing_20', epoch, dataset),
        r'        \midrule',
        r'        \multicolumn{5}{l}{\textit{Threshold $\tau=0.4$ (40\% Frozen)}} \\',
        format_row(r'PackNet (Magnitude)', 'packnet_40', epoch, dataset),
        format_row(r'Random Freezing', 'random_40', epoch, dataset),
        format_row(r'\textbf{P-Factor (Ours)}', 'freezing_40', epoch, dataset),
        r'        \midrule',
        r'        \multicolumn{5}{l}{\textit{Threshold $\tau=0.6$ (60\% Frozen)}} \\',
        format_row(r'PackNet (Magnitude)', 'packnet_60', epoch, dataset),
        format_row(r'Random Freezing', 'random_60', epoch, dataset),
        format_row(r'\textbf{P-Factor (Ours)}', 'freezing_60', epoch, dataset),
        r'        \midrule',
        r'        \multicolumn{5}{l}{\textit{Threshold $\tau=0.8$ (80\% Frozen)}} \\',
        format_row(r'PackNet (Magnitude)', 'packnet_80', epoch, dataset),
        format_row(r'Random Freezing', 'random_80', epoch, dataset),
        format_row(r'\textbf{P-Factor (Ours)}', 'freezing_80', epoch, dataset),
    ]
    return r"""\begin{table}[H]
    \caption{Comparison of Structural Engram Allocation Strategies (Split-MNIST, Epoch 3)}
    \begin{center}
    \resizebox{\columnwidth}{!}{
    \begin{tabular}{lcccc}
        \toprule
        \textbf{Configuration} & \textbf{Task A (Class-IL)} & \textbf{Task A (Task-IL)} & \textbf{Task B (Acc.)} & \textbf{Combined} \\
        \midrule
""" + "\n".join(rows) + r"""
        \bottomrule
    \end{tabular}}
    \label{tab:structural_comparison}
    \end{center}
\end{table}"""

def get_main_ablation_table():
    epoch = 3
    dataset = "Split-MNIST"
    rows = [
        r'        \multicolumn{5}{l}{\textit{Weight Scaling Ablation ($\tau=0.8$)}} \\',
        format_row(r'No-Scale', 'noscale_80', epoch, dataset),
        format_row(r'\textbf{P-Factor (With Scale)}', 'freezing_80', epoch, dataset),
        r'        \midrule',
        r'        \multicolumn{5}{l}{\textit{Novice Reset Strategy ($\tau=0.8$)}} \\',
        format_row(r'No-Reset', 'noreset_80', epoch, dataset),
        format_row(r'Reset-Zero', 'reset_zero_80', epoch, dataset),
        format_row(r'Reset-Scale', 'reset_scale_80', epoch, dataset),
        format_row(r'\textbf{Kaiming Reset (Ours)}', 'freezing_80', epoch, dataset),
    ]
    return r"""\begin{table}[H]
    \caption{Ablation Study: Forward Pass Scaling and Reset Mechanics (Split-MNIST, Epoch 3)}
    \begin{center}
    \resizebox{\columnwidth}{!}{
    \begin{tabular}{lcccc}
        \toprule
        \textbf{Configuration} & \textbf{Task A (Class-IL)} & \textbf{Task A (Task-IL)} & \textbf{Task B (Acc.)} & \textbf{Combined} \\
        \midrule
""" + "\n".join(rows) + r"""
        \bottomrule
    \end{tabular}}
    \label{tab:ablations}
    \end{center}
\end{table}"""

def get_nmnist_table():
    epoch = 3
    dataset = "Split-NMNIST"
    rows = [
        format_row(r'Fine-Tuning (Baseline)', 'cl_baseline', epoch, dataset),
        format_row(r'ER (Buffer=200) [Oracle]', 'er_200', epoch, dataset),
        format_row(r'EWC ($\lambda=10^6$)', 'ewc_1000000', epoch, dataset),
        format_row(r'SI ($c=10^6$)', 'si_1000000', epoch, dataset),
        r'        \midrule',
        r'        \multicolumn{5}{l}{\textit{Structural Isolation}} \\',
        format_row(r'PackNet (80\%)', 'packnet_80', epoch, dataset),
        format_row(r'Random Freezing (80\%)', 'random_80', epoch, dataset),
        format_row(r'\textbf{P-Factor $\tau=0.8$ (Ours)}', 'freezing_80', epoch, dataset),
    ]
    return r"""\begin{table}[H]
    \caption{Performance on Spatio-Temporal Event Streams (N-MNIST, Epoch 3)}
    \begin{center}
    \resizebox{\columnwidth}{!}{
    \begin{tabular}{lcccc}
        \toprule
        \textbf{Configuration} & \textbf{Task A (Class-IL)} & \textbf{Task A (Task-IL)} & \textbf{Task B (Acc.)} & \textbf{Combined} \\
        \midrule
""" + "\n".join(rows) + r"""
        \bottomrule
    \end{tabular}}
    \label{tab:nmnist_results}
    \end{center}
\end{table}"""

def replace_table(text, caption_start, replacement):
    # Find the table block starting from \begin{table}[H] to \end{table} that contains caption_start
    parts = text.split(r"\begin{table}[H]")
    for i in range(1, len(parts)):
        if caption_start in parts[i]:
            sub_parts = parts[i].split(r"\end{table}", 1)
            parts[i] = "\n" + replacement.replace(r"\begin{table}[H]", "").replace(r"\end{table}", "") + r"\end{table}" + sub_parts[1]
    return r"\begin{table}[H]".join(parts)

paper = replace_table(paper, r"\caption{Gradient-Based Baselines", get_main_gradient_table())
paper = replace_table(paper, r"\caption{Comparison of Structural Engram", get_main_structural_table())
paper = replace_table(paper, r"\caption{Ablation Study: Forward Pass Scaling", get_main_ablation_table())
paper = replace_table(paper, r"\caption{Performance on Spatio-Temporal Event", get_nmnist_table())

with open("Paper.tex", "w") as f:
    f.write(paper)

print("Tables updated with Task-IL and filled with all available values.")
