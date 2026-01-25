"""
common.py - Core Spiking Neural Network Components

This module implements the fundamental building blocks for spiking neural networks:
1. Surrogate gradient functions for backpropagation through discrete spikes
2. Leaky Integrate-and-Fire (LIF) neuron model

These components are shared across all SNN model variants (baseline, LTP/LTD).

References:
- Neftci, E. O., Mostafa, H., & Zenke, F. (2019). Surrogate Gradient Learning in 
  Spiking Neural Networks. IEEE Signal Processing Magazine.
"""

import torch
import torch.nn as nn


class SurrogateSpike(torch.autograd.Function):
    """
    Surrogate gradient function for backpropagation through spike generation.
    
    The forward pass produces binary spikes (0 or 1), but this discontinuity
    prevents standard backpropagation. The backward pass uses a smooth surrogate
    gradient (fast sigmoid derivative) to approximate the gradient flow.
    
    This approach is essential for training SNNs with gradient descent methods.
    """
    
    @staticmethod
    def forward(ctx, input, threshold):
        """
        Forward pass: Generate spikes where membrane potential exceeds threshold.
        
        Args:
            input (Tensor): Membrane potential values
            threshold (float): Spike firing threshold
            
        Returns:
            Tensor: Binary spike tensor (1 where input >= threshold, 0 otherwise)
        """
        # Save (input - threshold) for gradient computation
        ctx.save_for_backward(input - threshold)
        # Hard threshold: produce binary spikes
        return (input >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: Compute surrogate gradient using fast sigmoid derivative.
        
        Uses the derivative of a steep sigmoid function as the surrogate:
        grad = beta * exp(-beta * |x|) / (1 + exp(-beta * |x|))^2
        
        The beta parameter controls the steepness (higher = more spike-like).
        
        Args:
            grad_output (Tensor): Upstream gradients
            
        Returns:
            tuple: (gradient w.r.t. input, None for threshold)
        """
        (x,) = ctx.saved_tensors
        beta = 10.0  # Steepness of the surrogate gradient
        
        # Fast sigmoid surrogate gradient
        grad = beta * torch.exp(-beta * x.abs()) / ((1 + torch.exp(-beta * x.abs())) ** 2)
        
        return grad_output * grad, None


def spike_fn(input, threshold=1.0):
    """
    Convenience function for spike generation with surrogate gradients.
    
    Args:
        input (Tensor): Membrane potential values
        threshold (float): Spike firing threshold (default: 1.0)
        
    Returns:
        Tensor: Binary spike tensor
    """
    return SurrogateSpike.apply(input, threshold)


class LIFNeuron(nn.Module):
    """
    Leaky Integrate-and-Fire (LIF) Neuron Model.
    
    The LIF neuron is a biologically-inspired model that:
    1. Integrates incoming current into a membrane potential V
    2. Leaks potential over time (exponential decay with time constant tau)
    3. Fires a spike when V exceeds the threshold
    4. Resets V to zero after firing (soft reset)
    
    Dynamics:
        V[t] = alpha * V[t-1] + I[t]     (leaky integration)
        S[t] = 1 if V[t] >= threshold    (spike generation)
        V[t] = V[t] * (1 - S[t])         (soft reset after spike)
    
    where alpha = exp(-1/tau) is the leak factor.
    
    Attributes:
        V (Tensor): Membrane potential (state variable)
        threshold (float): Spike firing threshold
        alpha (float): Leak factor derived from tau
        size (int): Number of neurons in the layer
    """
    
    def __init__(self, size, threshold=1.0, tau=2.0):
        """
        Initialize LIF neuron layer.
        
        Args:
            size (int): Number of neurons in this layer
            threshold (float): Membrane potential threshold for spiking
            tau (float): Membrane time constant (higher = slower decay)
        """
        super().__init__()
        self.V = None  # Membrane potential (initialized on first forward pass)
        self.threshold = threshold
        self.alpha = torch.exp(torch.tensor(-1.0 / tau))  # Leak factor
        self.size = size

    def reset(self):
        """Reset membrane potential to None (re-initializes on next forward pass)."""
        self.V = None

    def forward(self, input):
        """
        Process one timestep of input current.
        
        Args:
            input (Tensor): Input current [batch_size, num_neurons]
            
        Returns:
            Tensor: Output spikes [batch_size, num_neurons]
        """
        # Initialize membrane potential if needed
        if self.V is None or self.V.shape != input.shape:
            self.V = torch.zeros_like(input)

        # Leaky integration: V = alpha * V_prev + I
        self.V = self.alpha * self.V + input

        # Spike generation using surrogate gradient
        S = spike_fn(self.V, self.threshold)

        # Soft reset: V = V * (1 - S), i.e., reset to 0 where spike occurred
        self.V = self.V * (1 - S)

        return S
