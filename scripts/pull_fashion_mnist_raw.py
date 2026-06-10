import os
import gzip
import numpy as np
import urllib.request
from scipy.io import savemat

def download_and_extract(url, extract_dir):
    filename = url.split('/')[-1]
    filepath = os.path.join(extract_dir, filename)
    
    if not os.path.exists(filepath):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filepath)
        
    with gzip.open(filepath, 'rb') as f:
        data = f.read()
    return data

def pull_fashion_mnist_raw(output_dir="FashionMNIST"):
    os.makedirs(output_dir, exist_ok=True)
    
    base_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
    files = {
        'train_images': base_url + "train-images-idx3-ubyte.gz",
        'train_labels': base_url + "train-labels-idx1-ubyte.gz",
        'test_images': base_url + "t10k-images-idx3-ubyte.gz",
        'test_labels': base_url + "t10k-labels-idx1-ubyte.gz"
    }
    
    print("Fetching raw data...")
    train_img_data = download_and_extract(files['train_images'], output_dir)
    train_lbl_data = download_and_extract(files['train_labels'], output_dir)
    test_img_data = download_and_extract(files['test_images'], output_dir)
    test_lbl_data = download_and_extract(files['test_labels'], output_dir)
    
    print("Parsing idx format...")
    # Images start at byte 16
    train_images = np.frombuffer(train_img_data, dtype=np.uint8, offset=16).reshape(-1, 784)
    test_images = np.frombuffer(test_img_data, dtype=np.uint8, offset=16).reshape(-1, 784)
    data = np.concatenate((train_images, test_images), axis=0).T # [784, 70000]
    
    # Labels start at byte 8
    train_labels = np.frombuffer(train_lbl_data, dtype=np.uint8, offset=8)
    test_labels = np.frombuffer(test_lbl_data, dtype=np.uint8, offset=8)
    labels = np.concatenate((train_labels, test_labels), axis=0).reshape(1, -1) # [1, 70000]
    
    mat_path = os.path.join(output_dir, "fashion-mnist-original.mat")
    savemat(mat_path, {'data': data, 'label': labels})
    
    print(f"Saved to {mat_path}")
    print(f"Data shape: {data.shape}")
    print(f"Label shape: {labels.shape}")

if __name__ == "__main__":
    pull_fashion_mnist_raw()
