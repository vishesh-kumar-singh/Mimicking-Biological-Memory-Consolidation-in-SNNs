# Mimicking Biological Memory Consolidation in Spiking Neural Networks

A research project exploring **Spiking Neural Networks (SNNs)** with a focus on **Long-Term Potentiation (LTP)** and **Long-Term Depression (LTD)** mechanisms. This work introduces the **P-Factor**—a biologically grounded utility metric that identifies predictive engrams to mitigate **catastrophic forgetting** in continual learning.

## Overview

Unlike standard magnitude-based pruning or global regularization approaches, this framework dynamically tracks utility during inference. By isolating and "freezing" high-utility engrams (Expert Neurons) and resetting low-utility neurons (Novice Neurons), this architecture establishes state-of-the-art structural plasticity.

We rigorously evaluate this mechanism across three diverse visual benchmarks:
- **Split-MNIST:** Standard sparse digit classification sequence.
- **Split-N-MNIST:** High-entropy spatio-temporal neuromorphic event streams.
- **5-Split Fashion-MNIST:** Dense spatial manifolds forming a 5-task sequence.

## Repository Structure

The codebase has been professionally organized into highly modular directories:

```
MNIST with SNN/
├── src/
│   ├── dataset.py              # Spike encoding & data wrappers
│   ├── utils.py                # Core helper functions
│   └── models/
│       ├── ltp_ltd.py              # SNN with dynamic P-factor (Neuron-Level)
│       ├── synaptic_p_factor.py    # Ideal Synaptic-Level P-Factor baseline
│       └── common.py               # Shared components (LIF neuron dynamics)
├── experiments/
│   ├── run_freezing.py             # Main P-Factor consolidation engine
│   ├── run_synaptic_baseline.py    # Synaptic-level consolidation experiment
│   ├── run_baseline_energy.py      # Energy/SynOps comparison suite
│   ├── run_packnet.py              # PackNet magnitude baseline
│   ├── run_ewc.py                  # Elastic Weight Consolidation baseline
│   ├── run_si.py                   # Synaptic Intelligence baseline
│   └── sweep_nmnist_exp.py         # Hyperparameter sensitivity sweeps
├── analysis/
│   ├── analyze_results.py              # Calculates Task-IL and Class-IL metrics
│   ├── analyze_energy.py               # Spiking energy efficiency calculator
│   ├── analyze_collateral.py           # Quantifies collateral locking in engrams
│   ├── analyze_sparsity_coverage.py    # Analyzes firing-rate sparsity distribution
│   └── measure_memory.py               # Calculates structural memory overhead
├── plotting/
│   ├── generate_paper_plots.py # Core visualization compiler
│   └── plot_structural_comparison.py # Engram allocation plotting
├── bash_scripts/
│   ├── run_all_split_mnist.sh  # Master execution suite for Split-MNIST
│   └── run_all_nmnist.sh       # Master execution suite for N-MNIST
├── utils/
│   ├── generate_data.py        # Generates Poisson spike-encoded tensors
│   └── update_paper.py         # Automated LaTeX compilation scripts
├── results/                    # Serialized execution histories (JSON)
├── checkpoints/                # PyTorch model weights
└── plots/                      # Generated publication-quality PDFs
```

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch (CUDA heavily recommended due to $\mathcal{O}(T)$ SNN time-step simulation)
- NumPy, SciPy, Matplotlib, Tqdm

```bash
# Clone the repository
git clone <repository-url>
cd "MNIST with SNN"

# Install dependencies
pip install -r requirements.txt
```

### Dataset Preparation

1. Download the original MNIST dataset (`mnist-original.mat`) and place it in `MNIST/`
2. Generate the 100-timestep Poisson spike-encoded tensors:
```bash
python utils/generate_data.py
```

## Execution Pipeline

We have automated the experimental execution. All core suites execute 5 random seeds to ensure statistical significance.

### 1. Run Complete Benchmark Suites
To reproduce the primary baseline comparisons (P-Factor vs. ER, EWC, SI, PackNet, Random):

```bash
# Execute the sparse data sequence (Split-MNIST)
bash bash_scripts/run_all_split_mnist.sh

# Execute the temporal event stream sequence (N-MNIST)
bash bash_scripts/run_all_nmnist.sh
bash bash_scripts/run_all_nmnist_baselines.sh

# Execute the highly dense 5-task sequence (5-Split Fashion-MNIST)
bash bash_scripts/run_fashion_mnist_multi_split.sh
```

### 2. Run Hyperparameter Heatmaps
To generate the $\alpha_{LTP}$ vs $\alpha_{LTD}$ sensitivity sweeps (demonstrating the bimodal capacity trade-off on dense data):

```bash
python experiments/sweep_mnist_exp.py
python experiments/sweep_nmnist_exp.py
```

### 3. Generate Paper Visualizations
All plotting scripts ingest the `results/` JSONs and output vector-graphic PDFs into the `plots/` folder:
```bash
python plotting/generate_paper_plots.py
python plotting/plot_structural_comparison.py
```

## Key Mechanisms

### 1. P-Factor Utility Tracking
Instead of relying on weight magnitude or Hessian diagonals, we modulate synaptic plasticity based on localized prediction success:
- **Correct prediction**: `P_new = P_old + α_LTP × (P_max - P_old)`
- **Wrong prediction**: `P_new = P_old - α_LTD`

### 2. Structural Consolidation
To prevent catastrophic forgetting:
1. **Engram Identification**: Post-Task A, neurons with a P-factor in the top $\tau$-percentile are identified as the memory engram.
2. **Gradient Isolation**: These expert neurons have their backward gradients masked during Task B.
3. **Novice Reset**: The remaining (plastic) neurons are reset and assigned to learn the new incoming spatial manifold.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please contact `visheshk24@iitk.ac.in` for discussion or collaboration.
