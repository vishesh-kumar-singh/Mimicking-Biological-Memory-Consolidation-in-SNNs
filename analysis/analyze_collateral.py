import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from models.ltp_ltd import SNNModelLTP_LTD

def analyze_collateral(checkpoint_path, threshold_percentile=0.8):
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint {checkpoint_path} not found.")
        return

    # Load model
    model = SNNModelLTP_LTD(input_size=784, hidden_size=1024, output_size=10)
    
    # Load state dict (mapping to cpu)
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict)
    
    # Extract P-factors and weights
    p_factors = model.layer1.P.detach()
    weights = model.layer1.linear.weight.detach()
    
    print(f"Total hidden neurons: {p_factors.size(0)}")
    print(f"Total incoming synapses per neuron: {weights.size(1)}")
    
    # Identify frozen neurons
    k = int(p_factors.size(0) * threshold_percentile)
    threshold = torch.kthvalue(p_factors, p_factors.size(0) - k + 1).values
    
    is_frozen = p_factors >= threshold
    frozen_idx = torch.where(is_frozen)[0]
    num_frozen = len(frozen_idx)
    
    print(f"\nAt tau={threshold_percentile}, {num_frozen} neurons are frozen.")
    
    # Extract the frozen weights
    frozen_weights = weights[frozen_idx, :]
    total_frozen_synapses = frozen_weights.numel()
    
    print(f"Total frozen synapses: {total_frozen_synapses}")
    
    # Analyze weight magnitudes
    abs_weights = torch.abs(frozen_weights)
    
    mean_w = abs_weights.mean().item()
    median_w = abs_weights.median().item()
    max_w = abs_weights.max().item()
    
    print(f"\nFrozen Weight Stats:")
    print(f"Mean absolute weight: {mean_w:.6f}")
    print(f"Median absolute weight: {median_w:.6f}")
    print(f"Max absolute weight: {max_w:.6f}")
    
    # Define thresholds
    thresholds = [0.001, 0.005, 0.01, 0.05, 0.1]
    print("\nCollateral Locking (Useless synapses frozen):")
    for t in thresholds:
        useless_count = (abs_weights < t).sum().item()
        pct = (useless_count / total_frozen_synapses) * 100
        print(f"  |w| < {t:<5}: {useless_count:6d} synapses ({pct:.2f}%)")
        
    # Analyze active vs inactive input pixels
    # Since this is MNIST, the edges are almost always zero. Let's see how many weights are just edge pixels.
    # We can approximate dead pixels by taking the mean of all weights for each input pixel across ALL neurons
    input_activity = torch.abs(weights).mean(dim=0)
    dead_inputs = (input_activity < 0.01).sum().item()
    print(f"\nApproximate dead input pixels (edges): {dead_inputs}/784")
    
    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(abs_weights.numpy().flatten(), bins=100, range=(0, 0.2), color='blue', alpha=0.7)
    plt.title(f'Distribution of Frozen Synapse Magnitudes (tau={threshold_percentile})')
    plt.xlabel('Absolute Weight Magnitude |w|')
    plt.ylabel('Frequency')
    plt.axvline(x=0.01, color='red', linestyle='--', label='Threshold |w| < 0.01')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(os.path.dirname(__file__), 'collateral_dist.png')
    plt.savefig(plot_path)
    print(f"\nSaved distribution plot to {plot_path}")

if __name__ == "__main__":
    cp = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'MNIST', 'seed_42_epochs_3_taskA.pt')
    analyze_collateral(cp, 0.8)
