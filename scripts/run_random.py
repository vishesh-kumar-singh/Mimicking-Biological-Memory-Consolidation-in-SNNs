"""
run_random.py - Random Neuron Freezing Control Experiment

This script implements a control experiment for comparison with P-factor freezing.
Instead of freezing neurons based on learned P-factors, it freezes a RANDOM
selection of neurons at the same percentile.

Purpose:
--------
This control demonstrates that the P-factor method's effectiveness is due to
WHICH neurons are frozen (those important for Task A), not simply HOW MANY.

Key Difference from run_freezing.py:
-----------------------------------
- run_freezing.py: Freezes neurons with highest P-factors (most important)
- run_random.py: Freezes randomly selected neurons (no importance criterion)

If random freezing performs worse than P-factor freezing at the same percentile,
it supports the hypothesis that P-factors meaningfully identify important neurons.

Usage:
------
    python scripts/run_random.py --epochs 3 --percentile 0.7 --runs 5
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import json
import argparse

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.dataset import SpikeMNISTDataset, NMNISTDatasetWrapper
from src.models import SNNModelBaseline

# =============================================================================
# Hyperparameters (same as other experiments for fair comparison)
# =============================================================================
BATCH_SIZE = 32
HIDDEN_SIZE = 1024
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "spike_mnist_dataset"


def evaluate(model, dataloader, device):
    """Evaluate model accuracy on a dataloader."""
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for spikes, labels in dataloader:
            spikes = spikes.to(device)
            labels = labels.to(device)
            model.reset()
            out = model(spikes)
            preds = out.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
    return total_correct / total_samples * 100 if total_samples > 0 else 0


def generate_random_mask_and_reset(model, threshold_percentile=0.7):
    """
    Generate freezing masks by RANDOMLY selecting neurons to freeze.
    
    Unlike P-factor freezing, this selects neurons uniformly at random,
    without regard to their importance or activity patterns.
    
    Args:
        model: Baseline SNN model
        threshold_percentile (float): Fraction of neurons to freeze
        
    Returns:
        dict: Gradient masks (0 for frozen, 1 for plastic)
    """
    masks = {}
    
    if hasattr(model, 'layer1'):
        # Get number of neurons in hidden layer
        num_neurons = model.layer1.linear.weight.shape[0]
        
        # Calculate number of neurons to FREEZE
        k_frozen = int(num_neurons * threshold_percentile)
        
        # RANDOM selection: shuffle indices and take first k
        perm = torch.randperm(num_neurons)
        frozen_indices = perm[:k_frozen]
        
        # Create mask: 1 for plastic (update), 0 for frozen (no update)
        mask1 = torch.ones(num_neurons, 1).to(DEVICE)
        mask1[frozen_indices] = 0.0
        masks['layer1'] = mask1
        
        # Reset plastic neurons (non-frozen) for Task B learning
        novice_indices = torch.where(mask1.squeeze() == 1)[0]
        if len(novice_indices) > 0:
            with torch.no_grad():
                nn.init.kaiming_uniform_(model.layer1.linear.weight[novice_indices], a=np.sqrt(5))

    if hasattr(model, 'layer2'):
        # Output layer: always freeze Task A outputs (0-4)
        mask2 = torch.ones(model.layer2.linear.weight.shape[0], 1).to(DEVICE)
        mask2[0:5] = 0.0
        masks['layer2'] = mask2
        
        # Reset Task B heads (5-9)
        with torch.no_grad():
            nn.init.kaiming_uniform_(model.layer2.linear.weight[5:], a=np.sqrt(5))
            
    return masks


def run_experiment(run_id, epochs, seed, percentile, data_dir="spike_mnist_dataset", is_nmnist=False):
    """
    Run random freezing control experiment.
    
    Same protocol as P-factor experiment but with random neuron selection.
    """
    print(f"\n{'='*40}")
    print(f"RUN {run_id+1} (Seed {seed}, Percentile {percentile}, Random Baseline)")
    print(f"{'='*40}")
    
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Load Data
    if is_nmnist:
        dataset_a = NMNISTDatasetWrapper(save_to=data_dir, train=True, target_digits=[0,1,2,3,4])
        dataset_b = NMNISTDatasetWrapper(save_to=data_dir, train=True, target_digits=[5,6,7,8,9])
        dataset_all = NMNISTDatasetWrapper(save_to=data_dir, train=True, target_digits=list(range(10)))
        input_dim = 2312
    else:
        spike_file = os.path.join(data_dir, "spike_trains_100ts.npy")
        label_file = os.path.join(data_dir, "labels.npy")
        if not os.path.exists(spike_file):
            print("Data not found.")
            return None
        dataset_a = SpikeMNISTDataset(spike_file, label_file, target_digits=[0,1,2,3,4])
        dataset_b = SpikeMNISTDataset(spike_file, label_file, target_digits=[5,6,7,8,9])
        dataset_all = SpikeMNISTDataset(spike_file, label_file, target_digits=list(range(10)))
        input_dim = 784
    
    def get_loader(ds):
        train_size = int(0.8 * len(ds))
        train_ds, test_ds = torch.utils.data.random_split(
            ds, [train_size, len(ds)-train_size], 
            generator=torch.Generator().manual_seed(seed)
        )
        return DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True), \
               DataLoader(test_ds, batch_size=BATCH_SIZE)

    train_a, test_a = get_loader(dataset_a)
    train_b, test_b = get_loader(dataset_b)
    _, test_all = get_loader(dataset_all)
    
    # NOTE: Using BASELINE SNN (no P-factors) since random selection doesn't use them
    model = SNNModelBaseline(input_size=input_dim, hidden_size=HIDDEN_SIZE).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    history = {
        "full_curve": [],
        "task_b": [],
        "eval_all": 0.0,
        "final_task_a": 0.0
    }

    # Phase 1: Train Task A (no P-factor updates)
    print("--- Phase 1: Training Task A ---")
    for epoch in range(epochs):
        model.train()
        total_correct = 0
        total_samples = 0
        
        pbar = tqdm(train_a, desc=f"Task A Epoch {epoch+1}")
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
            
            # NO P-FACTOR UPDATE (baseline model)
            
            pbar.set_postfix({"Loss": loss.item(), "Acc": total_correct/total_samples})
                
        acc = evaluate(model, test_a, DEVICE)
        history["full_curve"].append(acc)
        print(f"Epoch {epoch+1} Test Acc: {acc:.2f}%")

    # Phase 2: Random Freezing & Task B
    print(f"\n[System] Applying Random Freezing (Percentile {percentile})...")
    static_masks = generate_random_mask_and_reset(model, threshold_percentile=percentile)

    print("\n--- Phase 2: Training Task B ---")
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(epochs):
        model.train()
        total_correct = 0
        total_samples = 0
        
        pbar = tqdm(train_b, desc=f"Task B Epoch {epoch+1}")
        for s, l in pbar:
            s, l = s.to(DEVICE), l.to(DEVICE)
            optimizer.zero_grad()
            model.reset()
            out = model(s)
            loss = criterion(out, l)
            loss.backward()
            
            # Apply gradient masks
            if static_masks:
                if model.layer1.linear.weight.grad is not None:
                    model.layer1.linear.weight.grad.data.mul_(static_masks['layer1'])
                if model.layer2.linear.weight.grad is not None:
                    model.layer2.linear.weight.grad.data.mul_(static_masks['layer2'])
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            preds = out.argmax(dim=1)
            total_correct += (preds == l).sum().item()
            total_samples += l.size(0)
            
            pbar.set_postfix({"Loss": loss.item(), "Acc": total_correct/total_samples})
            
        # Evaluate retention and learning
        acc_retention = evaluate(model, test_a, DEVICE)
        history["full_curve"].append(acc_retention)
        print(f"Epoch {epoch+1} Task A Retention: {acc_retention:.2f}%")

        acc_b = evaluate(model, test_b, DEVICE)
        history["task_b"].append(acc_b)
        print(f"Epoch {epoch+1} Task B Accuracy: {acc_b:.2f}%")
        
    # Final Evaluation
    acc_all = evaluate(model, test_all, DEVICE)
    print(f"Combined Test Accuracy: {acc_all:.2f}%")
    
    history["eval_all"] = acc_all
    history["final_task_a"] = history["full_curve"][-1]
    
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Random Freezing Control Experiment")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs with different seeds")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs per task")
    parser.add_argument("--percentile", type=float, default=0.7, help="Freezing percentile")
    parser.add_argument("--data_dir", type=str, default="spike_mnist_dataset", help="Directory with spike data")
    parser.add_argument("--dataset_name", type=str, default="Split-MNIST", help="Name for results directory")
    parser.add_argument("--is_nmnist", action="store_true", help="Use NMNISTDatasetWrapper")
    args = parser.parse_args()
    
    from src.utils import load_legacy_json, parse_results_file, save_aggregated_results
    
    # Output Directory (same as other results)
    percentile_int = int(args.percentile * 100)
    output_dir = f"results/SNN/{args.dataset_name}/epochs_{args.epochs}"
    os.makedirs(output_dir, exist_ok=True)
    results_file = f"{output_dir}/random_{percentile_int}.json"
    
    # Check existing results
    histories = parse_results_file(results_file)
    
    existing_seeds = set()
    for h in histories:
        if 'seed' in h:
            existing_seeds.add(h['seed'])
            
    runs_needed = args.runs - len(histories)
    if runs_needed <= 0:
        print(f"Already have {len(histories)} runs (requested {args.runs}). Skipping...")
        sys.exit(0)
        
    print(f"Found {len(histories)} existing runs. Running {runs_needed} more...")

    runs_completed = 0
    current_seed = 42
    
    while runs_completed < runs_needed:
        if current_seed not in existing_seeds:
            hist = run_experiment(len(histories), args.epochs, current_seed, args.percentile, data_dir=args.data_dir, is_nmnist=args.is_nmnist)
            if hist:
                hist['seed'] = current_seed
                histories.append(hist)
                save_aggregated_results(results_file, histories)
                print(f"Saved result for seed {current_seed} to {results_file}")
            runs_completed += 1
        current_seed += 1
    
    if args.runs == 0 and histories:
         save_aggregated_results(results_file, histories)
         print(f"Aggregated results saved to {results_file}")
