import torch
from src.models.ltp_ltd import SNNModelLTP_LTD
from src.models.synaptic_p_factor import SynapticPFactorLIFNetwork
import os
import sys

# Ensure the project root is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def measure_tensor_memory(tensor: torch.Tensor, name: str):
    bytes_size = tensor.element_size() * tensor.nelement()
    kb_size = bytes_size / 1024
    mb_size = kb_size / 1024
    print(f"{name}: {tensor.size()}")
    print(f"  -> {bytes_size} bytes ({kb_size:.2f} KB | {mb_size:.2f} MB)")
    return bytes_size

def main():
    # Instantiate models
    print("Measuring P-Factor Buffer Memory Overhead...\n")
    
    neuron_model = SNNModelLTP_LTD(input_size=784, hidden_size=1024, output_size=10)
    synaptic_model = SynapticPFactorLIFNetwork(input_size=784, hidden_size=1024, output_size=10)
    
    # Measure layer 1 buffers
    print("--- Layer 1 ---")
    neuron_mem = measure_tensor_memory(neuron_model.layer1.P, "Neuron-Level P-Factor Buffer")
    synapse_mem = measure_tensor_memory(synaptic_model.layer1.P, "Synaptic-Level P-Factor Buffer")
    
    ratio = synapse_mem / neuron_mem
    print(f"\nRatio: Synaptic method uses {ratio:.1f}x more memory for state tracking.\n")

if __name__ == '__main__':
    main()
