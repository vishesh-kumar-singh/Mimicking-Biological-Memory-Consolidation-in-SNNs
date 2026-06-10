"""
ltp_ltd.py - Long-Term Potentiation/Depression SNN Model

This module implements the core contribution of this research: an SNN with
biologically-inspired plasticity mechanisms for mitigating catastrophic forgetting.

Key Concepts:
-------------
1. P-Factor: Each neuron maintains a "permanence" factor (P) that modulates
   its synaptic weights. P represents the neuron's importance for learned tasks.
   
2. LTP (Long-Term Potentiation): Neurons that fire during CORRECT predictions
   have their P increased, strengthening their contribution.
   Formula: P_new = P_old + alpha * (P_max - P_old)
   
3. LTD (Long-Term Depression): Neurons that fire during WRONG predictions
   have their P decreased, weakening unreliable pathways.
   Formula: P_new = P_old - alpha

4. Weight Modulation: Effective weights are computed as W_eff = W * (1 + P)
   - P > 0: Amplifies synaptic strength
   - P < 0: Attenuates synaptic strength
   - P = 0: No modulation (baseline behavior)

5. Neuron Freezing: After Task A training, neurons with high P-factors are
   frozen (gradients masked) during Task B training to preserve Task A knowledge.

Biological Inspiration:
-----------------------
LTP and LTD are real biological processes in the brain that strengthen or
weaken synaptic connections based on neural activity patterns. Our P-factor
mechanism is a simplified computational analog.

References:
-----------
- Hebb, D. O. (1949). The Organization of Behavior.
- Bliss, T. V., & Lømo, T. (1973). Long-lasting potentiation of synaptic
  transmission in the dentate area of the anaesthetized rabbit.
"""

import torch
import torch.nn as nn
from .common import LIFNeuron


def update_p_factor_ltp(model, correct_indices, alpha=0.01, p_max=1.0):
    """
    Apply Long-Term Potentiation to neurons involved in correct predictions.
    
    For each sample that was correctly classified, identify neurons that fired
    and increase their P-factor. This strengthens synapses that contributed to
    successful pattern recognition.
    
    The update rule uses exponential saturation toward p_max:
        P_new = P_old + alpha * (P_max - P_old)
    
    This ensures P asymptotically approaches but never exceeds p_max.
    
    Args:
        model (SNNModelLTP_LTD): The model containing P-factors
        correct_indices (Tensor): Batch indices of correctly predicted samples
        alpha (float): Learning rate for P-factor updates
        p_max (float): Maximum P-factor value (saturation limit)
    """
    if len(correct_indices) == 0:
        return

    with torch.no_grad():
        # Layer 1: Find neurons that fired for any correct sample
        fired1 = model.layer1_fired[correct_indices]  # [num_correct, hidden_size]
        neurons_to_update_1 = fired1.any(dim=0)  # [hidden_size]
        model.layer1.P[neurons_to_update_1] += alpha * (p_max - model.layer1.P[neurons_to_update_1])
        
        # Layer 2: Same for output layer
        fired2 = model.layer2_fired[correct_indices]
        neurons_to_update_2 = fired2.any(dim=0)
        model.layer2.P[neurons_to_update_2] += alpha * (p_max - model.layer2.P[neurons_to_update_2])
        
        # Clamp to ensure bounds (safety measure)
        model.layer1.P.clamp_(max=p_max)
        model.layer2.P.clamp_(max=p_max)


def update_p_factor_ltd(model, wrong_indices, alpha=0.01, p_min=-1.0):
    """
    Apply Long-Term Depression to neurons involved in incorrect predictions.
    
    For each sample that was misclassified, identify neurons that fired
    and decrease their P-factor. This weakens synapses that contributed to
    incorrect pattern recognition.
    
    The update rule is linear decay:
        P_new = P_old - alpha
    
    This allows P to decrease and potentially become negative, which
    attenuates the effective synaptic weight.
    
    Args:
        model (SNNModelLTP_LTD): The model containing P-factors
        wrong_indices (Tensor): Batch indices of incorrectly predicted samples
        alpha (float): Learning rate for P-factor updates
        p_min (float): Minimum P-factor value (floor limit)
    """
    if len(wrong_indices) == 0:
        return

    with torch.no_grad():
        # Layer 1: Find neurons that fired for any wrong sample
        fired1 = model.layer1_fired[wrong_indices]
        neurons_to_update_1 = fired1.any(dim=0)
        model.layer1.P[neurons_to_update_1] -= alpha
        
        # Layer 2: Same for output layer
        fired2 = model.layer2_fired[wrong_indices]
        neurons_to_update_2 = fired2.any(dim=0)
        model.layer2.P[neurons_to_update_2] -= alpha
        
        # Clamp to ensure bounds (prevent unbounded decay)
        model.layer1.P.clamp_(min=p_min)
        model.layer2.P.clamp_(min=p_min)


class SNNLayerLTP_LTD(nn.Module):
    """
    Single layer of the LTP/LTD SNN with P-factor modulation.
    
    This layer extends the baseline by adding:
    1. A P-factor buffer for each neuron (initialized to 0)
    2. Weight modulation during forward pass: W_eff = W * (1 + P)
    
    The P-factor acts as a multiplicative gain on synaptic weights,
    allowing the network to remember which neurons are important.
    
    Attributes:
        linear (nn.Linear): Synaptic weight matrix
        neuron (LIFNeuron): LIF neuron dynamics
        P (Tensor): P-factor buffer [out_dim], registered as non-trainable buffer
    """
    
    def __init__(self, in_dim, out_dim, threshold=1.0, tau=2.0, scale_weights=True):
        """
        Initialize LTP/LTD layer.
        
        Args:
            in_dim (int): Input dimension (presynaptic neurons)
            out_dim (int): Output dimension (postsynaptic neurons)
            threshold (float): Spike threshold for LIF neurons
            tau (float): Membrane time constant
            scale_weights (bool): If True, modulate weights by (1+P). If False, use raw weights.
        """
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.neuron = LIFNeuron(size=out_dim, threshold=threshold, tau=tau)
        self.scale_weights = scale_weights
        
        # P-factor: registered as buffer (not a parameter, not trained by optimizer)
        # Initialized to 0, updated by LTP/LTD rules
        self.register_buffer('P', torch.zeros(out_dim))
        
    def forward(self, input_spikes):
        """
        Forward pass with optional P-factor weight modulation.
        
        This applies the formula: W_eff = W * (1 + P)
        where P is the plasticity factor modulated by LTP/LTD.
        
        Args:
            input_spikes (Tensor): Binary spike input [batch_size, in_dim]
            
        Returns:
            Tensor: Output spikes [batch_size, out_dim]
        """
        if self.scale_weights:
            effective_weight = self.linear.weight * (1 + self.P.unsqueeze(1))
            I = torch.nn.functional.linear(input_spikes, effective_weight)
        else:
            I = self.linear(input_spikes)
        
        spikes = self.neuron(I)
        return spikes


class SNNModelLTP_LTD(nn.Module):
    """
    Complete 2-layer SNN with LTP/LTD plasticity mechanisms.
    
    This is the main model for our continual learning experiments. It extends
    the baseline SNN with:
    
    1. P-factor modulation in each layer
    2. Spike tracking for LTP/LTD updates (layer1_fired, layer2_fired)
    3. Support for selective neuron freezing based on P-values
    
    Architecture:
        Input (784) -> Hidden (256/1024) -> Output (10)
        
    Training Procedure:
        1. Forward pass: compute output spikes with P-modulated weights
        2. Loss computation: CrossEntropyLoss on spike counts
        3. Backward pass: update weights via backprop
        4. P-factor update: apply LTP/LTD based on prediction correctness
        
    Continual Learning:
        After Task A training:
        - Neurons with high P are frozen (gradients masked)
        - Task B training only updates low-P neurons
        - This preserves Task A knowledge in high-P neurons
    """
    
    def __init__(self, input_size=784, hidden_size=256, output_size=10, time_steps=100, threshold=1.0, scale_weights=True):
        """
        Initialize the LTP/LTD SNN.
        
        Args:
            input_size (int): Input dimension (784 for flattened MNIST)
            hidden_size (int): Hidden layer neurons
            output_size (int): Output classes (10 for MNIST)
            time_steps (int): Number of simulation timesteps
            threshold (float): Spike threshold for neurons
            scale_weights (bool): If True, modulate weights by (1+P). If False, use raw weights.
        """
        super().__init__()
        self.time_steps = time_steps
        self.layer1 = SNNLayerLTP_LTD(input_size, hidden_size, threshold=threshold, scale_weights=scale_weights)
        self.layer2 = SNNLayerLTP_LTD(hidden_size, output_size, threshold=threshold, scale_weights=scale_weights)

    def reset(self):
        """Reset all neuron membrane potentials between samples."""
        self.layer1.neuron.reset()
        self.layer2.neuron.reset()

    def forward(self, x):
        """
        Forward pass: process spike train and track firing neurons.
        
        In addition to computing output spike counts, this method tracks
        which neurons fired during the forward pass. This information is
        stored in layer1_fired and layer2_fired for subsequent LTP/LTD updates.
        
        Args:
            x (Tensor): Input spike trains [batch_size, time_steps, input_size]
            
        Returns:
            Tensor: Output spike counts [batch_size, output_size]
        """
        batch_size = x.size(0)
        output_spike_count = torch.zeros((batch_size, 10), device=x.device)
        
        # Track which neurons fired for each sample (for LTP/LTD)
        self.layer1_fired = torch.zeros((batch_size, self.layer1.neuron.size), device=x.device, dtype=torch.bool)
        self.layer2_fired = torch.zeros((batch_size, self.layer2.neuron.size), device=x.device, dtype=torch.bool)

        # Simulate network dynamics over time
        for t in range(self.time_steps):
            input_t = x[:, t, :]
            
            spikes1 = self.layer1(input_t)
            spikes2 = self.layer2(spikes1)
            
            output_spike_count += spikes2
            
            # Update firing records (OR accumulation over time)
            self.layer1_fired = self.layer1_fired | (spikes1 > 0)
            self.layer2_fired = self.layer2_fired | (spikes2 > 0)

        return output_spike_count


def update_p_factor_combined(model, preds, labels, alpha_ltp=0.01, alpha_ltd=0.01, p_max=1.0):
    """
    Convenience function to apply both LTP and LTD in a single call.
    
    This is the main function called after each batch during training.
    It separates correct and incorrect predictions and applies the
    appropriate plasticity rule to each.
    
    Args:
        model (SNNModelLTP_LTD): The model to update
        preds (Tensor): Predicted class labels [batch_size]
        labels (Tensor): Ground truth labels [batch_size]
        alpha_ltp (float): Learning rate for LTP
        alpha_ltd (float): Learning rate for LTD
        p_max (float): Maximum P-factor value
    """
    # Separate correct and incorrect predictions
    correct_mask = (preds == labels)
    wrong_mask = (preds != labels)
    
    correct_indices = torch.where(correct_mask)[0]
    wrong_indices = torch.where(wrong_mask)[0]
    
    # Apply LTP to neurons that contributed to correct predictions
    if len(correct_indices) > 0:
        update_p_factor_ltp(model, correct_indices, alpha=alpha_ltp, p_max=p_max)
        
    # Apply LTD to neurons that contributed to incorrect predictions
    if len(wrong_indices) > 0:
        update_p_factor_ltd(model, wrong_indices, alpha=alpha_ltd)
