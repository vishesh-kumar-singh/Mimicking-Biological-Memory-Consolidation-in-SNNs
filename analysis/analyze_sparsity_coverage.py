import sys
import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.dataset import SpikeMNISTDataset
from src.models import SNNModelLTP_LTD, update_p_factor_combined

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HIDDEN_SIZE = 1024
TAU = 0.8  # Freezing 80%

def main():
    print(f"Running on {DEVICE}")
    
    data_dir = "spike_mnist_dataset"
    spike_file = os.path.join(data_dir, "spike_trains_100ts.npy")
    label_file = os.path.join(data_dir, "labels.npy")
    
    if not os.path.exists(spike_file):
        print(f"Data not found at {spike_file}")
        return
        
    print("Loading datasets...")
    dataset_a = SpikeMNISTDataset(spike_file, label_file, target_digits=[0,1,2,3,4])
    
    # We want a train/test split to train for 1 epoch, then evaluate sparsity on test.
    train_size = int(0.8 * len(dataset_a))
    train_ds, test_ds = torch.utils.data.random_split(
        dataset_a, [train_size, len(dataset_a)-train_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    
    model = SNNModelLTP_LTD(input_size=784, hidden_size=HIDDEN_SIZE).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Train for 1 epoch to get meaningful weights and P-factors
    print("Training for 1 epoch to establish P-factors and weights...")
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        model.reset()
        
        output_spikes = model(images)
        loss = criterion(output_spikes, labels)
        loss.backward()
        optimizer.step()
        
        preds = output_spikes.argmax(dim=1)
        correct = (preds == labels)
        
        # Update P-factor
        update_p_factor_combined(model, preds, labels, alpha_ltp=0.01, alpha_ltd=0.01)

    print("Evaluating firing rates on validation set...")
    model.eval()
    total_spikes = torch.zeros(HIDDEN_SIZE, device=DEVICE)
    total_samples = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            model.reset()
            _ = model(images)
            
            # layer1_fired is a boolean tensor of shape [batch_size, hidden_size]
            total_spikes += model.layer1_fired.sum(dim=0).float()
            total_samples += images.size(0)
            
    avg_firing_rate = (total_spikes / total_samples).cpu().numpy()
    
    # Sort firing rates descending
    sorted_rates = np.sort(avg_firing_rate)[::-1]
    
    # Plot the distribution
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_rates, linewidth=2)
    plt.title("Task A Firing-Rate Distribution (Hidden Layer)", fontsize=24)
    plt.xlabel("Neuron Rank (Most Active to Least Active)", fontsize=20)
    plt.ylabel("Average Spikes per Sample", fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True)
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/firing_rate_distribution.pdf")
    print("Saved firing rate distribution plot to plots/firing_rate_distribution.pdf")
    
    # Define "Task-relevant" neurons (e.g., top 20% most active)
    top_20_percent = int(HIDDEN_SIZE * 0.2)
    active_indices = np.argsort(avg_firing_rate)[::-1][:top_20_percent]
    
    print(f"\nDefined 'Task-relevant' neurons as the top {top_20_percent} most active neurons.")
    
    # Get P-factors
    p_factors = model.layer1.P.cpu().detach().numpy()
    
    # P-Factor Freezing at tau = 0.8 (freezing 80% of neurons = 819 neurons)
    num_frozen = int(HIDDEN_SIZE * TAU)
    p_factor_frozen_indices = np.argsort(p_factors)[::-1][:num_frozen]
    
    # Coverage calculation
    p_factor_coverage = len(set(active_indices).intersection(set(p_factor_frozen_indices))) / top_20_percent
    
    # Random Freezing Coverage (expected value is exactly TAU, i.e., 80%)
    random_coverages = []
    for _ in range(100):
        random_frozen_indices = np.random.choice(HIDDEN_SIZE, num_frozen, replace=False)
        random_cov = len(set(active_indices).intersection(set(random_frozen_indices))) / top_20_percent
        random_coverages.append(random_cov)
    random_coverage = np.mean(random_coverages)
    
    print(f"\n--- Coverage Results at tau = {TAU} ({num_frozen} neurons frozen) ---")
    print(f"P-Factor Coverage of Task-Relevant Neurons: {p_factor_coverage*100:.2f}%")
    print(f"Random Freezing Coverage of Task-Relevant Neurons: {random_coverage*100:.2f}%")

if __name__ == "__main__":
    main()
