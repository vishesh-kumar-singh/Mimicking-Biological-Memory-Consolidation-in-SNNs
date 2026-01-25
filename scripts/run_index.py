"""
run_index.py - Index-Based Neuron Freezing Control Experiment

This script implements another control experiment for comparison with P-factor
freezing. Instead of using P-factors or random selection, it freezes neurons
based purely on their INDEX POSITION in the layer.

Purpose:
--------
This control tests whether P-factor freezing's effectiveness comes from:
1. The specific neurons identified by P-factors (our hypothesis)
2. Simply having ANY deterministic selection criterion

If index freezing performs worse than P-factor freezing, it further supports
that P-factors meaningfully identify task-important neurons.

Key Difference:
--------------
- run_freezing.py: Freezes neurons with highest P-factors (learned importance)
- run_random.py: Freezes randomly selected neurons
- run_index.py: Freezes first n% neurons by index (arbitrary but deterministic)

Usage:
------
    python scripts/run_index.py --epochs 3 --percentile 0.7 --runs 5
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

from src.dataset import SpikeMNISTDataset
from src.models import SNNModelBaseline

# =============================================================================
# Hyperparameters
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


def generate_index_mask_and_reset(model, threshold_percentile=0.7):
    """
    Generate freezing masks based on neuron INDEX position.
    
    Freezes the first n% of neurons by their index:
    - E.g., 70% freezing on 1024 neurons: freeze indices 0-716
    
    This is a purely structural criterion with no relation to learned
    importance or activity patterns.
    
    Args:
        model: Baseline SNN model
        threshold_percentile (float): Fraction of neurons to freeze
        
    Returns:
        dict: Gradient masks (0 for frozen, 1 for plastic)
    """
    masks = {}
    
    if hasattr(model, 'layer1'):
        num_neurons = model.layer1.linear.weight.shape[0]
        
        # Calculate number to freeze
        k_frozen = int(num_neurons * threshold_percentile)
        
        # INDEX-BASED: Freeze first k neurons (indices 0 to k-1)
        mask1 = torch.ones(num_neurons, 1).to(DEVICE)
        mask1[:k_frozen] = 0.0  # First k neurons are frozen
        masks['layer1'] = mask1
        
        # Reset plastic neurons (indices k to end)
        plastic_indices = torch.arange(k_frozen, num_neurons)
        if len(plastic_indices) > 0:
            with torch.no_grad():
                nn.init.kaiming_uniform_(model.layer1.linear.weight[plastic_indices], a=np.sqrt(5))

    if hasattr(model, 'layer2'):
        # Output layer: always freeze Task A outputs (0-4)
        mask2 = torch.ones(model.layer2.linear.weight.shape[0], 1).to(DEVICE)
        mask2[0:5] = 0.0
        masks['layer2'] = mask2
        
        # Reset Task B heads (5-9)
        with torch.no_grad():
            nn.init.kaiming_uniform_(model.layer2.linear.weight[5:], a=np.sqrt(5))
            
    return masks


def run_experiment(run_id, epochs, seed, percentile):
    """
    Run index-based freezing control experiment.
    
    Same protocol as other experiments but freezes neurons by index.
    """
    print(f"\n{'='*40}")
    print(f"RUN {run_id+1} (Seed {seed}, Percentile {percentile}, Index Freezing)")
    print(f"{'='*40}")
    
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Load Data
    spike_file = os.path.join(DATA_DIR, "spike_trains_100ts.npy")
    label_file = os.path.join(DATA_DIR, "labels.npy")
    if not os.path.exists(spike_file):
        print("Data not found.")
        return None

    # Datasets
    dataset_a = SpikeMNISTDataset(spike_file, label_file, target_digits=[0,1,2,3,4])
    dataset_b = SpikeMNISTDataset(spike_file, label_file, target_digits=[5,6,7,8,9])
    dataset_all = SpikeMNISTDataset(spike_file, label_file, target_digits=list(range(10)))
    
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
    
    # Using BASELINE SNN
    model = SNNModelBaseline(hidden_size=HIDDEN_SIZE).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    history = {
        "full_curve": [],
        "task_b": [],
        "eval_all": 0.0,
        "final_task_a": 0.0
    }

    # Phase 1: Train Task A
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
            
            pbar.set_postfix({"Loss": loss.item(), "Acc": total_correct/total_samples})
                
        acc = evaluate(model, test_a, DEVICE)
        history["full_curve"].append(acc)
        print(f"Epoch {epoch+1} Test Acc: {acc:.2f}%")

    # Phase 2: Index Freezing & Task B
    print(f"\n[System] Applying Index Freezing (Percentile {percentile})...")
    static_masks = generate_index_mask_and_reset(model, threshold_percentile=percentile)

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
    parser = argparse.ArgumentParser(description="Index Freezing Control Experiment")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs with different seeds")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs per task")
    parser.add_argument("--percentile", type=float, default=0.7, help="Freezing percentile")
    args = parser.parse_args()
    
    from src.utils import load_legacy_json, parse_results_file, save_aggregated_results
    
    # Output Directory (same as other results)
    percentile_int = int(args.percentile * 100)
    output_dir = f"results/results_epochs_{args.epochs}"
    os.makedirs(output_dir, exist_ok=True)
    results_file = f"{output_dir}/index_{percentile_int}.json"
    
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
            hist = run_experiment(len(histories), args.epochs, current_seed, args.percentile)
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
