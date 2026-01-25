"""
baseline.py - Baseline Spiking Neural Network Model

This module implements a standard SNN without any plasticity mechanisms.
It serves as the baseline (control) for comparing against LTP/LTD variants.

The baseline SNN uses:
- Standard LIF neurons with fixed synaptic weights
- CrossEntropyLoss for training
- No P-factor or weight modulation

In continual learning experiments, this baseline demonstrates catastrophic
forgetting when trained sequentially on Task A then Task B.
"""

import torch
import torch.nn as nn
from .common import LIFNeuron


class SNNLayerBaseline(nn.Module):
    """
    Single layer of the baseline SNN.
    
    Consists of:
    - Linear transformation (synaptic weights)
    - LIF neuron dynamics
    
    No plasticity mechanisms are included - weights are only updated
    via standard backpropagation during training.
    """
    
    def __init__(self, in_dim, out_dim, threshold=1.0, tau=2.0):
        """
        Initialize the SNN layer.
        
        Args:
            in_dim (int): Input dimension (presynaptic neurons)
            out_dim (int): Output dimension (postsynaptic neurons)
            threshold (float): Spike threshold for LIF neurons
            tau (float): Membrane time constant for leak
        """
        super().__init__()
        # Linear layer without bias (biologically, synapses don't have offsets)
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.neuron = LIFNeuron(size=out_dim, threshold=threshold, tau=tau)

    def forward(self, input_spikes):
        """
        Process input spikes through the layer.
        
        Args:
            input_spikes (Tensor): Binary spike input [batch_size, in_dim]
            
        Returns:
            Tensor: Output spikes [batch_size, out_dim]
        """
        # Synaptic integration: weighted sum of input spikes
        I = self.linear(input_spikes)
        # Neuronal dynamics: produce output spikes
        spikes = self.neuron(I)
        return spikes


class SNNModelBaseline(nn.Module):
    """
    Complete 2-layer baseline SNN for MNIST classification.
    
    Architecture:
        Input (784) -> Hidden (256/1024) -> Output (10)
        
    The model processes spike trains over T timesteps and counts
    output spikes per class for classification.
    
    This model exhibits catastrophic forgetting in continual learning
    scenarios and serves as the control baseline.
    """
    
    def __init__(self, input_size=784, hidden_size=256, output_size=10, time_steps=100):
        """
        Initialize the baseline SNN.
        
        Args:
            input_size (int): Input dimension (784 for flattened MNIST)
            hidden_size (int): Hidden layer neurons (256 or 1024)
            output_size (int): Output classes (10 for MNIST)
            time_steps (int): Number of simulation timesteps
        """
        super().__init__()
        self.time_steps = time_steps
        self.layer1 = SNNLayerBaseline(input_size, hidden_size)
        self.layer2 = SNNLayerBaseline(hidden_size, output_size)

    def reset(self):
        """Reset all neuron membrane potentials between samples."""
        self.layer1.neuron.reset()
        self.layer2.neuron.reset()

    def forward(self, x):
        """
        Forward pass: process spike train over all timesteps.
        
        Args:
            x (Tensor): Input spike trains [batch_size, time_steps, input_size]
            
        Returns:
            Tensor: Output spike counts [batch_size, output_size]
                    Used with argmax for classification
        """
        batch_size = x.size(0)
        # Accumulate output spikes across all timesteps
        output_spike_count = torch.zeros((batch_size, 10), device=x.device)

        # Simulate network dynamics over time
        for t in range(self.time_steps):
            input_t = x[:, t, :]  # Input at timestep t
            spikes1 = self.layer1(input_t)  # Hidden layer spikes
            spikes2 = self.layer2(spikes1)  # Output layer spikes
            output_spike_count += spikes2  # Accumulate output spikes

        return output_spike_count
