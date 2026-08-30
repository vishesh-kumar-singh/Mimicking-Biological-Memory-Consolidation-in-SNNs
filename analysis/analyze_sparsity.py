import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataset import SpikeMNISTDataset
from src.models import SNNModelLTP_LTD

BATCH_SIZE = 32
HIDDEN_SIZE = 1024
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "spike_mnist_dataset"
EPOCHS = 3
SEED = 42

def main():
    print("="*50)
    print("SNN Sparsity Analysis")
    print("="*50)
    
    # Set seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    spike_file = os.path.join(DATA_DIR, "spike_trains_100ts.npy")
    label_file = os.path.join(DATA_DIR, "labels.npy")
    
    if not os.path.exists(spike_file):
        print("Data not found. Please ensure dataset is generated.")
        return

    dataset_a = SpikeMNISTDataset(spike_file, label_file, target_digits=[0,1,2,3,4])
    
    train_size = int(0.8 * len(dataset_a))
    train_ds, test_ds = torch.utils.data.random_split(
        dataset_a, [train_size, len(dataset_a)-train_size], 
        generator=torch.Generator().manual_seed(SEED)
    )
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    
    model = SNNModelLTP_LTD(hidden_size=HIDDEN_SIZE).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    print(f"\nTraining on Task A for {EPOCHS} epochs to establish memory trace...")
    
    for epoch in range(EPOCHS):
        model.train()
        total_correct = 0
        total_samples = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for s, l in pbar:
            s, l = s.to(DEVICE), l.to(DEVICE)
            
            optimizer.zero_grad()
            model.reset()
            out = model(s)
            loss = criterion(out, l)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            preds = out.argmax(dim=1)
            total_correct += (preds == l).sum().item()
            total_samples += l.size(0)
            
            pbar.set_postfix({"Loss": loss.item(), "Acc": total_correct/total_samples})
            
    print("\nTraining complete. Measuring hidden layer firing rates on test set...")
    
    model.eval()
    all_firing_rates = []
    
    with torch.no_grad():
        for s, l in tqdm(test_loader, desc="Testing"):
            s = s.to(DEVICE)
            model.reset()
            _ = model(s)
            
            # layer1_fired is a boolean tensor of shape [batch_size, hidden_size]
            # It records if a neuron fired AT LEAST ONCE during the 100 timesteps for that sample
            batch_fired = model.layer1_fired.float() 
            all_firing_rates.append(batch_fired)
            
    # Concatenate all batches [total_samples, hidden_size]
    all_firing_rates = torch.cat(all_firing_rates, dim=0)
    
    # Compute the mean firing rate per neuron across the entire test set
    # i.e. fraction of test images that caused this neuron to fire
    per_neuron_rates = all_firing_rates.mean(dim=0).cpu().numpy()
    
    print("\n" + "="*50)
    print("Sparsity Analysis Results")
    print("="*50)
    
    thresholds = [0.01, 0.05, 0.10, 0.20, 0.50]
    total_neurons = HIDDEN_SIZE
    
    print(f"Total Hidden Neurons: {total_neurons}")
    print(f"Mean Firing Rate across all neurons: {per_neuron_rates.mean()*100:.2f}%")
    print(f"Median Firing Rate: {np.median(per_neuron_rates)*100:.2f}%")
    
    print("\nNeurons exceeding firing thresholds:")
    for t in thresholds:
        active = np.sum(per_neuron_rates > t)
        print(f"  > {t*100:2.0f}% active: {active:4d} neurons ({active/total_neurons*100:5.1f}%)")
        
    silent = np.sum(per_neuron_rates == 0)
    print(f"\nCompletely silent neurons (0.0%): {silent} ({silent/total_neurons*100:.1f}%)")
    
    # Theoretical Expected Overlap with Random Freezing
    print("\n--- Theoretical Impact of Random Freezing ---")
    active_thresh = 0.05
    active_count = np.sum(per_neuron_rates > active_thresh)
    print(f"Assume 'True Engram' consists of neurons firing > {active_thresh*100}% of the time ({active_count} neurons).")
    
    freeze_fractions = [0.4, 0.6, 0.8]
    for frac in freeze_fractions:
        num_frozen = int(total_neurons * frac)
        # Expected number of active neurons randomly frozen
        expected_active_frozen = active_count * frac
        print(f"If randomly freezing {frac*100:.0f}% ({num_frozen} neurons):")
        print(f"  -> Expected active neurons protected: {expected_active_frozen:.0f} out of {active_count} ({frac*100:.0f}%)")
        print(f"  -> This mathematically explains why Random Freezing achieves ~{frac*100:.0f}% retention without any intelligence!")

    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(per_neuron_rates, bins=50, color='skyblue', edgecolor='black')
    plt.title('Hidden Neuron Firing Rate Distribution (Task A)', fontsize=24)
    plt.xlabel('Fraction of Test Samples where Neuron Fired', fontsize=20)
    plt.ylabel('Number of Neurons', fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(axis='y', alpha=0.75)
    
    plt.axvline(x=0.05, color='r', linestyle='--', label='5% Threshold')
    plt.legend(fontsize=18)
    
    out_dir = "results/SNN/Split-MNIST"
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "sparsity_histogram.png"), dpi=300, bbox_inches='tight')
    print(f"\nHistogram saved to {out_dir}/sparsity_histogram.png")

if __name__ == "__main__":
    main()
