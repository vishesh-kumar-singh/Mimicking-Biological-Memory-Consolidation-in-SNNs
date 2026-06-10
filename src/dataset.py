"""
dataset.py - Spike Data Generation and Loading Utilities

This module provides utilities for converting MNIST images to spike trains
using Poisson encoding, and loading the pre-generated spike data efficiently
using memory-mapped files.

Key Components:
--------------
1. poisson_spike_encoding: Converts pixel intensities to temporal spike patterns
2. SpikeMNISTDataset: PyTorch Dataset for loading spike-encoded MNIST
3. generate_spike_data: One-time generation of spike data from raw MNIST

Poisson Encoding:
-----------------
Each pixel intensity p ∈ [0, 1] determines the probability of a spike at
each timestep. Over T timesteps, pixels with higher intensity produce more
spikes on average. This is a simple but effective rate coding scheme.

Memory Mapping:
--------------
Spike data is stored as memory-mapped files for efficient loading:
- Avoids loading entire dataset into RAM
- Enables fast random access
- Critical for large datasets and limited memory

References:
-----------
Diehl, P. U., & Cook, M. (2015). Unsupervised learning of digit recognition
using spike-timing-dependent plasticity. Frontiers in Computational Neuroscience.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat
from tqdm import tqdm


def poisson_spike_encoding(images, time_steps=100):
    """
    Convert grayscale images to spike trains using Poisson (rate) encoding.
    
    For each pixel, the intensity determines the probability of a spike
    at each timestep. This creates a temporal representation where brighter
    pixels produce more spikes over time.
    
    Mathematical formulation:
        P(spike at time t) = pixel_intensity
        spike[t] = 1 if random() < pixel_intensity else 0
    
    Args:
        images (np.ndarray): Normalized images [num_samples, 784], values in [0, 1]
        time_steps (int): Number of timesteps for spike encoding
        
    Returns:
        np.ndarray: Binary spike trains [num_samples, time_steps, 784]
    """
    # Generate random values and compare with pixel intensities
    # Broadcasting: images[:, None, :] is [N, 1, 784], random is [N, T, 784]
    return (np.random.rand(images.shape[0], time_steps, 784) < images[:, None, :]).astype(np.uint8)


class SpikeMNISTDataset(Dataset):
    """
    PyTorch Dataset for loading pre-generated spike-encoded MNIST data.
    
    Uses memory-mapped files for efficient loading of large spike datasets.
    Supports filtering by digit class for continual learning experiments
    (e.g., loading only digits 0-4 for Task A).
    
    The dataset expects pre-generated spike files created by generate_spike_data().
    
    Attributes:
        spikes (np.memmap): Memory-mapped spike data [N, T, 784]
        labels (np.memmap): Memory-mapped labels [N]
        indices (list): Filtered indices for target_digits subset
    """
    
    def __init__(self, spike_file, label_file, num_samples=70000, time_steps=100, input_dim=784, target_digits=None):
        """
        Initialize the spike MNIST dataset.
        
        Args:
            spike_file (str): Path to spike trains file (.npy memmap)
            label_file (str): Path to labels file (.npy memmap)
            num_samples (int): Total samples in dataset (70000 for full MNIST)
            time_steps (int): Number of timesteps per sample
            input_dim (int): Input dimension (784 for MNIST)
            target_digits (list): Optional list of digit classes to include.
                                  If None, all digits are included.
                                  Example: [0,1,2,3,4] for Task A
        """
        self.spike_file = spike_file
        self.label_file = label_file
        
        if not os.path.exists(spike_file) or not os.path.exists(label_file):
            raise FileNotFoundError(f"Spike file {spike_file} or label file {label_file} not found. Please generate them first.")

        # Memory-mapped loading for efficiency
        self.spikes = np.memmap(spike_file, dtype=np.uint8, mode="r", shape=(num_samples, time_steps, input_dim))
        self.labels = np.memmap(label_file, dtype=np.uint8, mode="r", shape=(num_samples,))
        
        # Filter indices if specific digits are requested (for continual learning)
        if target_digits is not None:
            self.indices = [i for i, label in enumerate(self.labels) if label in target_digits]
        else:
            self.indices = list(range(len(self.labels)))

    def __len__(self):
        """Return number of samples in the (filtered) dataset."""
        return len(self.indices)

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        Args:
            idx (int): Index into the filtered dataset
            
        Returns:
            tuple: (spike_tensor, label_tensor)
                - spike_tensor: [time_steps, 784] float32
                - label_tensor: scalar long
        """
        real_idx = self.indices[idx]
        spike_tensor = torch.tensor(self.spikes[real_idx], dtype=torch.float32)
        label_tensor = torch.tensor(self.labels[real_idx], dtype=torch.long)
        return spike_tensor, label_tensor


def generate_spike_data(mat_file, output_dir, time_steps=100, chunk_size=1000):
    """
    Generate spike-encoded MNIST data from raw .mat file.
    
    This is a one-time preprocessing step that:
    1. Loads raw MNIST from .mat format
    2. Normalizes pixel values to [0, 1]
    3. Applies Poisson spike encoding
    4. Saves as memory-mapped files for efficient loading
    
    Processing is done in chunks to manage memory usage.
    
    Args:
        mat_file (str): Path to MNIST .mat file (e.g., 'MNIST/mnist-original.mat')
        output_dir (str): Directory to save spike files
        time_steps (int): Number of timesteps for spike encoding
        chunk_size (int): Samples to process per batch (for memory efficiency)
        
    Output Files:
        - {output_dir}/spike_trains_{time_steps}ts.npy: Spike data [70000, T, 784]
        - {output_dir}/labels.npy: Labels [70000]
    """
    if not os.path.exists(mat_file):
        raise FileNotFoundError(f"MNIST mat file not found at {mat_file}")
        
    os.makedirs(output_dir, exist_ok=True)
    
    # Load raw MNIST data
    print(f"Loading {mat_file}...")
    mnist = loadmat(mat_file)
    mnist_data = mnist["data"].T  # Transpose to [N, 784]
    mnist_label = mnist["label"][0]  # Flatten to [N]
    
    print(f"Data shape: {mnist_data.shape}")
    print(f"Label shape: {mnist_label.shape}")
    print(f"Unique labels in source: {np.unique(mnist_label)}")
    
    num_samples = mnist_data.shape[0]
    input_dim = 784
    
    # Output file paths
    spike_path = os.path.join(output_dir, f"spike_trains_{time_steps}ts.npy")
    label_path = os.path.join(output_dir, "labels.npy")
    
    # Create memory-mapped output files
    print(f"Generating spike data to {output_dir}...")
    spike_memmap = np.memmap(spike_path, dtype=np.uint8, mode='w+', shape=(num_samples, time_steps, input_dim))
    label_memmap = np.memmap(label_path, dtype=np.uint8, mode='w+', shape=(num_samples,))
    
    # Process in chunks for memory efficiency
    for i in tqdm(range(0, num_samples, chunk_size)):
        # Normalize chunk to [0, 1]
        chunk = mnist_data[i:i+chunk_size] / 255.0
        label_chunk = mnist_label[i:i+chunk_size]
        
        # Generate spikes using Poisson encoding
        spike_chunk = poisson_spike_encoding(chunk, time_steps).astype(np.uint8)
        
        # Write to memory-mapped file
        current_chunk_size = spike_chunk.shape[0]
        spike_memmap[i:i+current_chunk_size] = spike_chunk
        label_memmap[i:i+current_chunk_size] = label_chunk
        
    # Flush to disk
    spike_memmap.flush()
    label_memmap.flush()
    
    # Verification: check that labels were written correctly
    written_labels = np.memmap(label_path, dtype=np.uint8, mode='r', shape=(num_samples,))
    print(f"Unique labels in written file: {np.unique(written_labels)}")
    
    print("Generation complete.")

import torch
import tonic
import tonic.transforms as transforms

class NMNISTDatasetWrapper(torch.utils.data.Dataset):
    """
    Wrapper for tonic's N-MNIST dataset for continual learning.
    Transforms event streams into dense spike frames of shape (time_steps, 2312)
    where 2312 = 34 * 34 * 2 (W * H * Polarity).
    """
    def __init__(self, save_to='./data', train=True, time_steps=100, target_digits=None):
        sensor_size = tonic.datasets.NMNIST.sensor_size
        # Transform events to frames: (time_steps, 2, 34, 34)
        transform = transforms.Compose([
            transforms.ToFrame(sensor_size=sensor_size, n_time_bins=time_steps)
        ])
        
        self.dataset = tonic.datasets.NMNIST(save_to=save_to, train=train, transform=transform)
        
        # Filter indices for specific tasks (e.g. 0-4 for Task A)
        if target_digits is not None:
            self.indices = [i for i, target in enumerate(self.dataset.targets) if target in target_digits]
        else:
            self.indices = list(range(len(self.dataset)))
            
    def __len__(self):
        return len(self.indices)
        
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        events, target = self.dataset[real_idx]
        
        # events shape: (time_steps, 2, 34, 34)
        # We need to flatten it to (time_steps, 2312)
        # and cast to float32 since the model expects float/float32 spikes
        spikes = torch.tensor(events, dtype=torch.float32).view(events.shape[0], -1)
        
        return spikes, target
