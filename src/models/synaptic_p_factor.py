import torch
import torch.nn as nn
from .common import LIFNeuron

def update_synaptic_p_factor_ltp(model, correct_indices, alpha=0.01, p_max=1.0, static_masks=None):
    if len(correct_indices) == 0:
        return

    with torch.no_grad():
        # Layer 1
        post_fired1 = model.layer1_post_fired[correct_indices] # [correct_batch, out_dim]
        pre_fired1 = model.layer1_pre_fired[correct_indices]   # [correct_batch, in_dim]
        
        # We want to update P where both pre and post fired in the same batch
        # For each sample in correct_batch, synapse (i,j) fired if post_i and pre_j fired.
        # Active synapses over the batch: shape [out_dim, in_dim]
        active_synapses1 = torch.einsum('bi,bj->ij', post_fired1.float(), pre_fired1.float()) > 0
        
        if static_masks is not None and 'layer1' in static_masks:
            active_synapses1 = active_synapses1 & static_masks['layer1'].bool()
            
        model.layer1.P[active_synapses1] += alpha * (p_max - model.layer1.P[active_synapses1])
        
        # Layer 2
        post_fired2 = model.layer2_post_fired[correct_indices] # [correct_batch, out_dim]
        pre_fired2 = model.layer2_pre_fired[correct_indices]   # [correct_batch, in_dim]
        
        active_synapses2 = torch.einsum('bi,bj->ij', post_fired2.float(), pre_fired2.float()) > 0
        
        if static_masks is not None and 'layer2' in static_masks:
            active_synapses2 = active_synapses2 & static_masks['layer2'].bool()
            
        model.layer2.P[active_synapses2] += alpha * (p_max - model.layer2.P[active_synapses2])
        
        model.layer1.P.clamp_(max=p_max)
        model.layer2.P.clamp_(max=p_max)


def update_synaptic_p_factor_ltd(model, wrong_indices, alpha=0.01, p_min=-1.0, static_masks=None):
    if len(wrong_indices) == 0:
        return

    with torch.no_grad():
        # Layer 1
        post_fired1 = model.layer1_post_fired[wrong_indices] 
        pre_fired1 = model.layer1_pre_fired[wrong_indices]   
        
        active_synapses1 = torch.einsum('bi,bj->ij', post_fired1.float(), pre_fired1.float()) > 0
        
        if static_masks is not None and 'layer1' in static_masks:
            active_synapses1 = active_synapses1 & static_masks['layer1'].bool()
            
        model.layer1.P[active_synapses1] -= alpha
        
        # Layer 2
        post_fired2 = model.layer2_post_fired[wrong_indices] 
        pre_fired2 = model.layer2_pre_fired[wrong_indices]   
        
        active_synapses2 = torch.einsum('bi,bj->ij', post_fired2.float(), pre_fired2.float()) > 0
        
        if static_masks is not None and 'layer2' in static_masks:
            active_synapses2 = active_synapses2 & static_masks['layer2'].bool()
            
        model.layer2.P[active_synapses2] -= alpha
        
        model.layer1.P.clamp_(min=p_min)
        model.layer2.P.clamp_(min=p_min)


class SynapticSNNLayerLTP_LTD(nn.Module):
    def __init__(self, in_dim, out_dim, threshold=1.0, tau=2.0, scale_weights=True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.neuron = LIFNeuron(size=out_dim, threshold=threshold, tau=tau)
        self.scale_weights = scale_weights
        
        # P-factor is now [out_dim, in_dim]
        self.register_buffer('P', torch.zeros(out_dim, in_dim))
        
    def forward(self, input_spikes):
        if self.scale_weights:
            effective_weight = self.linear.weight * (1 + self.P)
            I = torch.nn.functional.linear(input_spikes, effective_weight)
        else:
            I = self.linear(input_spikes)
        
        spikes = self.neuron(I)
        return spikes


class SynapticPFactorLIFNetwork(nn.Module):
    def __init__(self, input_size=784, hidden_size=256, output_size=10, time_steps=100, threshold=1.0, scale_weights=True):
        super().__init__()
        self.time_steps = time_steps
        self.layer1 = SynapticSNNLayerLTP_LTD(input_size, hidden_size, threshold=threshold, scale_weights=scale_weights)
        self.layer2 = SynapticSNNLayerLTP_LTD(hidden_size, output_size, threshold=threshold, scale_weights=scale_weights)

    def reset(self):
        self.layer1.neuron.reset()
        self.layer2.neuron.reset()

    def forward(self, x):
        batch_size = x.size(0)
        output_spike_count = torch.zeros((batch_size, 10), device=x.device)
        
        self.layer1_post_fired = torch.zeros((batch_size, self.layer1.neuron.size), device=x.device, dtype=torch.bool)
        self.layer1_pre_fired = torch.zeros((batch_size, x.size(-1)), device=x.device, dtype=torch.bool)
        
        self.layer2_post_fired = torch.zeros((batch_size, self.layer2.neuron.size), device=x.device, dtype=torch.bool)
        self.layer2_pre_fired = torch.zeros((batch_size, self.layer1.neuron.size), device=x.device, dtype=torch.bool)

        for t in range(self.time_steps):
            input_t = x[:, t, :]
            
            spikes1 = self.layer1(input_t)
            spikes2 = self.layer2(spikes1)
            
            output_spike_count += spikes2
            
            self.layer1_pre_fired = self.layer1_pre_fired | (input_t > 0)
            self.layer1_post_fired = self.layer1_post_fired | (spikes1 > 0)
            
            self.layer2_pre_fired = self.layer2_pre_fired | (spikes1 > 0)
            self.layer2_post_fired = self.layer2_post_fired | (spikes2 > 0)

        return output_spike_count


def update_synaptic_p_factor_combined(model, preds, labels, alpha_ltp=0.01, alpha_ltd=0.01, p_max=1.0, static_masks=None):
    correct_mask = (preds == labels)
    wrong_mask = (preds != labels)
    
    correct_indices = torch.where(correct_mask)[0]
    wrong_indices = torch.where(wrong_mask)[0]
    
    if len(correct_indices) > 0:
        update_synaptic_p_factor_ltp(model, correct_indices, alpha=alpha_ltp, p_max=p_max, static_masks=static_masks)
        
    if len(wrong_indices) > 0:
        update_synaptic_p_factor_ltd(model, wrong_indices, alpha=alpha_ltd, static_masks=static_masks)
