with open("Paper.tex", "r") as f:
    text = f.read()

# Locate the target section to replace
start_marker = r"\subsection{Energy and Computational Cost Analysis}"
end_marker = r"\section{Conclusion}"

if start_marker in text and end_marker in text:
    pre = text.split(start_marker)[0]
    post = text.split(end_marker)[1]
    
    new_section = r"""\subsection{Energy and Computational Cost Analysis}

In Spiking Neural Networks (SNNs), energy efficiency is primarily derived from activation sparsity. Unlike traditional Artificial Neural Networks (ANNs) that perform dense, high-precision floating-point Multiply-Accumulate (MAC) operations for every neuron connection at every forward pass, SNNs utilize simple integer Accumulate (AC) operations---referred to as Synaptic Operations (SynOps)---only when a neuron emits a discrete spike. In standard neuromorphic hardware benchmarks, a 32-bit MAC operation consumes approximately $3.1\text{ pJ}$, whereas a sparse AC operation consumes only $\approx 0.1\text{ pJ}$. The goal of our P-Factor consolidation method is to prevent catastrophic forgetting while preserving this inherent sparsity, keeping the total SynOps computationally cheap compared to baseline continual learning models that often devolve into dense, chaotic firing patterns when adapting to new tasks.

\subsubsection{Energy Analysis: Split-MNIST}
\begin{table}[H]
    \caption{Split-MNIST: Energy Efficiency and Sparsity Comparison (Epoch 1)}
    \begin{center}
    \resizebox{\columnwidth}{!}{
    \begin{tabular}{lcccc}
        \toprule
        \textbf{Method} & \textbf{L1 Sparsity (\%)} & \textbf{SynOps / Sample} & \textbf{Est. Energy ($\mu$J)} \\
        \midrule
        Standard ANN (Dense) & 0.0\% & -- (MACs: 813K) & $\sim 2.52$ \\
        SNN Baseline (Fine-Tuning) & -- & -- & -- \\
        EWC ($\lambda=10^6$) & -- & -- & -- \\
        SI ($c=10^6$) & -- & -- & -- \\
        PackNet (80\%) & -- & -- & -- \\
        \textbf{P-Factor $\tau=0.4$ (Ours)} & 91.61\% & 10,577,963 & $1.05$ \\
        \bottomrule
    \end{tabular}}
    \label{tab:energy_mnist}
    \end{center}
\end{table}

\subsubsection{Energy Analysis: Split-N-MNIST}
\begin{table}[H]
    \caption{Split-N-MNIST: Energy Efficiency and Sparsity Comparison}
    \begin{center}
    \resizebox{\columnwidth}{!}{
    \begin{tabular}{lcccc}
        \toprule
        \textbf{Method} & \textbf{L1 Sparsity (\%)} & \textbf{SynOps / Sample} & \textbf{Est. Energy ($\mu$J)} \\
        \midrule
        Standard ANN (Dense) & 0.0\% & -- & -- \\
        SNN Baseline (Fine-Tuning) & -- & -- & -- \\
        \textbf{P-Factor (Ours)} & -- & -- & -- \\
        \bottomrule
    \end{tabular}}
    \label{tab:energy_nmnist}
    \end{center}
\end{table}

\subsubsection{Energy Analysis: 5-Split-Fashion-MNIST}
\begin{table}[H]
    \caption{5-Split-Fashion-MNIST: Energy Efficiency and Sparsity Comparison}
    \begin{center}
    \resizebox{\columnwidth}{!}{
    \begin{tabular}{lcccc}
        \toprule
        \textbf{Method} & \textbf{L1 Sparsity (\%)} & \textbf{SynOps / Sample} & \textbf{Est. Energy ($\mu$J)} \\
        \midrule
        Standard ANN (Dense) & 0.0\% & -- & -- \\
        SNN Baseline (Fine-Tuning) & -- & -- & -- \\
        \textbf{P-Factor (Ours)} & -- & -- & -- \\
        \bottomrule
    \end{tabular}}
    \label{tab:energy_fmnist}
    \end{center}
\end{table}

\subsubsection{Memory and Computational Complexity}
\begin{table}[H]
    \caption{Computational Overhead and Method Complexity Comparison}
    \begin{center}
    \resizebox{\columnwidth}{!}{
    \begin{tabular}{lccc}
        \toprule
        \textbf{Method} & \textbf{Memory Overhead} & \textbf{Compute Overhead} & \textbf{Complexity} \\
        \midrule
        Fine-Tuning & None & None & $O(1)$ \\
        ER (Buffer=200) & High (Raw Data) & Forward/Backward on Buffer & $O(D_{mem})$ \\
        EWC & High (Float Matrix) & Hessian/Fisher Matrix Calc. & $O(W)$ \\
        SI & High (Float Matrix) & Path Integral Accumulation & $O(W)$ \\
        \textbf{P-Factor} & \textbf{Low (Float Vector)} & \textbf{Negligible (Thresholding)} & \textbf{$O(N)$} \\
        \bottomrule
    \end{tabular}}
    \label{tab:energy_analysis_complexity}
    \end{center}
\end{table}

\section{Conclusion}"""

    new_text = pre + new_section + post
    with open("Paper.tex", "w") as f:
        f.write(new_text)
    print("Energy analysis added successfully.")
else:
    print("Could not find start/end markers in Paper.tex")
