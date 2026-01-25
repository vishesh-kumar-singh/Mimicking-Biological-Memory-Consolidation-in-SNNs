import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from scipy.io import loadmat
import os
import sys
from tqdm import tqdm

                                                  
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.dataset import poisson_spike_encoding

def generate_data():
    mat_file = "MNIST/mnist-original.mat"
    output_dir = "spike_mnist_dataset"
    time_steps = 100
    
    if not os.path.exists(mat_file):
        print(f"Error: {mat_file} not found.")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading {mat_file}...")
    mnist = loadmat(mat_file)
    mnist_data = mnist["data"].T
    mnist_label = mnist["label"][0]
    
    print(f"Data shape: {mnist_data.shape}")
    print(f"Label shape: {mnist_label.shape}")
    print(f"Unique labels: {np.unique(mnist_label)}")
    
    num_samples = mnist_data.shape[0]
    input_dim = 784
    
    spike_path = os.path.join(output_dir, f"spike_trains_{time_steps}ts.npy")
    label_path = os.path.join(output_dir, "labels.npy")
    
    print(f"Generating spike data to {output_dir}...")
    spike_memmap = np.memmap(spike_path, dtype=np.uint8, mode='w+', shape=(num_samples, time_steps, input_dim))
    label_memmap = np.memmap(label_path, dtype=np.uint8, mode='w+', shape=(num_samples,))
    
    chunk_size = 1000
    for i in tqdm(range(0, num_samples, chunk_size)):
        chunk = mnist_data[i:i+chunk_size] / 255.0
        label_chunk = mnist_label[i:i+chunk_size]
        
        spike_chunk = poisson_spike_encoding(chunk, time_steps).astype(np.uint8)
        
        current_chunk_size = spike_chunk.shape[0]
        spike_memmap[i:i+current_chunk_size] = spike_chunk
        label_memmap[i:i+current_chunk_size] = label_chunk
        
    spike_memmap.flush()
    label_memmap.flush()
    
            
    written_labels = np.memmap(label_path, dtype=np.uint8, mode='r', shape=(num_samples,))
    print(f"Written Unique labels: {np.unique(written_labels)}")
    
    if len(np.unique(written_labels)) > 1:
        print("SUCCESS: Data generation looks correct.")
    else:
        print("FAILURE: Only one label found!")

if __name__ == "__main__":
    generate_data()
