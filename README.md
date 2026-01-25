# MNIST with Spiking Neural Networks (SNNs)

A research project exploring **Spiking Neural Networks (SNNs)** on the MNIST dataset with a focus on **Long-Term Potentiation (LTP)** and **Long-Term Depression (LTD)** mechanisms to mitigate **catastrophic forgetting** in continual learning scenarios.

## Overview

This repository implements and compares different SNN architectures for MNIST digit classification:

- **Baseline SNN**: A standard spiking neural network using Leaky Integrate-and-Fire (LIF) neurons
- **LTP/LTD SNN**: An enhanced SNN with P-factor modulation based on prediction outcomes:
  - **LTP (Long-Term Potentiation)**: Strengthens synapses for neurons that contribute to correct predictions
  - **LTD (Long-Term Depression)**: Weakens synapses for neurons that fire during incorrect predictions

The project focuses on **Split MNIST** experiments where the model is trained on digits 0-4 (Task A) first, then on digits 5-9 (Task B), testing retention of Task A knowledge.

## Project Structure

```
MNIST with SNN/
├── MNIST/                      # Original MNIST dataset (.mat format)
├── spike_mnist_dataset/        # Generated spike-encoded dataset
├── src/
│   ├── dataset.py              # Spike encoding & data loading utilities
│   ├── utils.py                # Helper functions
│   └── models/
│       ├── baseline.py         # Baseline SNN implementation
│       ├── ltp_ltd.py          # LTP/LTD SNN with P-factor mechanism
│       └── common.py           # Shared components (LIF neuron)
├── scripts/
│   ├── generate_data.py        # Generate spike-encoded MNIST data
│   ├── run_exp.py              # Master experiment orchestrator
│   ├── run_baseline.py         # Baseline continual learning experiments
│   ├── run_freezing.py         # P-factor based neuron freezing experiments
│   ├── run_random.py           # Random freezing baseline experiments
│   ├── plot_results.py         # Generate result visualizations
│   ├── analyze_results.py      # Statistical analysis of results
│   └── generate_paper_plots.py # Publication-quality figures
├── results/                    # Experiment results (JSON)
├── plots/                      # Generated visualizations
└── README.md                   # This file
```

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch (with CUDA support recommended)
- NumPy, SciPy, tqdm

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd "MNIST with SNN"

# Install dependencies
pip install torch numpy scipy tqdm matplotlib
```

### Prepare the Dataset

1. Place the MNIST dataset in `.mat` format at `MNIST/mnist-original.mat`

2. Generate spike-encoded data:
```bash
python scripts/generate_data.py
```

This creates Poisson spike-encoded MNIST data with 100 time steps per sample.

## Running Experiments

### Quick Start

Run the full experiment suite:
```bash
python scripts/run_exp.py --runs 5
```

This runs:
- Baseline SNN experiments
- P-factor freezing experiments (40%, 50%, 60%, 70%, 80% thresholds)
- Random freezing experiments (as control)

For all epoch configurations (1-5 epochs per task).

### Individual Experiments

**Baseline Continual Learning:**
```bash
python scripts/run_baseline.py --runs 5 --epochs 3
```

**P-factor Freezing:**
```bash
python scripts/run_freezing.py --runs 5 --epochs 3 --percentile 0.7
```

**Random Freezing (Control):**
```bash
python scripts/run_random.py --runs 5 --epochs 3 --percentile 0.7
```

### Experiment Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--runs` | Number of runs with different seeds | 5 |
| `--epochs` | Training epochs per task | 5 |
| `--percentile` | Threshold percentile for freezing | 0.7 |

## Key Concepts

### Spike Encoding
Images are converted to spike trains using **Poisson encoding** where pixel intensity determines firing probability.

### P-Factor Mechanism
Each neuron has a P-factor that modulates its synaptic weights:
- **Correct prediction**: `P_new = P_old + α × (P_max - P_old)`  (LTP)
- **Wrong prediction**: `P_new = P_old - α`  (LTD)

The effective weight becomes: `W_eff = W × (1 + P)`

### Neuron Freezing
To prevent catastrophic forgetting:
1. After Task A training, neurons with P-factor above a threshold are identified
2. Their weights are frozen during Task B training
3. This preserves Task A knowledge while allowing plasticity for Task B

## Analyzing Results

Generate plots from results:
```bash
python scripts/plot_results.py
python scripts/generate_paper_plots.py
```

Results are saved as JSON files in `results/` and plots in `plots/`.

## Model Architecture

| Layer | Dimensions | Description |
|-------|------------|-------------|
| Input | 784 | Flattened 28×28 MNIST images |
| Hidden | 1024 | LIF neurons with P-factor modulation |
| Output | 10 | Spike counts for each digit class |

**Hyperparameters:**
- Batch Size: 32
- Learning Rate: 1e-3
- Time Steps: 100
- Optimizer: Adam

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please mail me at visheshk24@iitk.ac.in for any further discussion or contribution to the work.
