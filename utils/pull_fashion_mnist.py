import os
import numpy as np
import torchvision
from scipy.io import savemat

def pull_fashion_mnist(output_dir="FashionMNIST"):
    os.makedirs(output_dir, exist_ok=True)
    
    print("Downloading Fashion MNIST...")
    train_set = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True)
    test_set = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True)
    
    print("Converting to .mat format...")
    # combine train and test
    train_data = train_set.data.numpy().reshape(-1, 784)
    test_data = test_set.data.numpy().reshape(-1, 784)
    data = np.concatenate((train_data, test_data), axis=0).T # [784, 70000]
    
    train_labels = train_set.targets.numpy()
    test_labels = test_set.targets.numpy()
    labels = np.concatenate((train_labels, test_labels), axis=0).reshape(1, -1) # [1, 70000]
    
    mat_path = os.path.join(output_dir, "fashion-mnist-original.mat")
    savemat(mat_path, {'data': data, 'label': labels})
    
    print(f"Saved to {mat_path}")
    print(f"Data shape: {data.shape}")
    print(f"Label shape: {labels.shape}")

if __name__ == "__main__":
    pull_fashion_mnist()
