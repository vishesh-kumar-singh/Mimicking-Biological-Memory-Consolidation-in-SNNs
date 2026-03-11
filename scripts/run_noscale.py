"""
run_noscale.py - P-Factor Tracking Without Weight Scaling (Ablation Study)

This ablation study uses the LTP/LTD P-factor mechanism to TRACK neuron
importance and IDENTIFY engrams for freezing, but does NOT scale weights
by (1 + P) during the forward pass. This isolates the benefit of
P-factor-based engram identification from the weight modulation effect.

Usage:
    python scripts/run_noscale.py --epochs 3 --percentile 0.6 --runs 5
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

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.dataset import SpikeMNISTDataset
from src.models import SNNModelLTP_LTD, update_p_factor_combined

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


def generate_static_mask_and_reset(model, threshold_percentile=0.7):
    """
    Generate freezing masks using P-factor values for engram identification.
    
    Uses the same P-factor thresholding as run_freezing.py to identify
    important neurons, but this model doesn't use P-factors for weight
    scaling (the ablation being tested).
    
    Args:
        model: LTP/LTD SNN model (with scale_weights=False)
        threshold_percentile (float): Fraction of neurons to freeze
        
    Returns:
        dict: Gradient masks (0 for frozen, 1 for plastic)
    """
    masks = {}
    if hasattr(model, 'layer1') and hasattr(model.layer1, 'P'):
        p1 = model.layer1.P.detach()
        k = int(p1.size(0) * threshold_percentile)
        threshold1 = torch.kthvalue(p1, p1.size(0) - k + 1).values
        # Freeze neurons with P >= threshold (engram neurons)
        mask1 = (p1 < threshold1).float().unsqueeze(1).to(p1.device)
        masks['layer1'] = mask1

        # Reset plastic (non-frozen) neurons for Task B learning
        novice_indices = torch.where(mask1.squeeze() == 1)[0]
        if len(novice_indices) > 0:
            with torch.no_grad():
                nn.init.kaiming_uniform_(model.layer1.linear.weight[novice_indices], a=np.sqrt(5))

    if hasattr(model, 'layer2'):
        # Output layer: always freeze Task A outputs (0-4)
        mask2 = torch.ones(model.layer2.linear.weight.shape[0], 1).to(p1.device)
        mask2[0:5] = 0.0
        masks['layer2'] = mask2

        # Reset Task B heads (5-9)
        if hasattr(model.layer2, 'P'):
            with torch.no_grad():
                model.layer2.P[5:] = 0.0
                nn.init.kaiming_uniform_(model.layer2.linear.weight[5:], a=np.sqrt(5))
    return masks


def run_experiment(run_id, epochs, seed, percentile):
    """
    Run a single no-scale ablation experiment.
    
    Uses SNNModelLTP_LTD with scale_weights=False: P-factors are tracked
    via LTP/LTD and used for engram identification, but NOT used to
    modulate weights in the forward pass.
    
    Args:
        run_id (int): Run identifier for logging
        epochs (int): Training epochs per task
        seed (int): Random seed for reproducibility
        percentile (float): Fraction of neurons to freeze
        
    Returns:
        dict: History with full_curve, task_b, eval_all, final_task_a
    """
    print(f"\n{'='*40}")
    print(f"RUN {run_id+1} (Seed {seed}, Percentile {percentile}, NoScale)")
    print(f"{'='*40}")

    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Load spike-encoded MNIST data
    spike_file = os.path.join(DATA_DIR, "spike_trains_100ts.npy")
    label_file = os.path.join(DATA_DIR, "labels.npy")
    if not os.path.exists(spike_file):
        print("Data not found.")
        return None

    # Continual learning datasets: Task A (0-4), Task B (5-9), All (0-9)
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

    # KEY DIFFERENCE: scale_weights=False
    # P-factors are still tracked via LTP/LTD but NOT used to modulate weights
    model = SNNModelLTP_LTD(hidden_size=HIDDEN_SIZE, scale_weights=False).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    history = {
        "full_curve": [],
        "task_b": [],
        "eval_all": 0.0,
        "final_task_a": 0.0
    }

    # Phase 1: Train on Task A (digits 0-4) with P-factor tracking
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

            # P-factors are still updated for engram identification
            update_p_factor_combined(model, preds, l, alpha_ltp=0.01)

            pbar.set_postfix({"Loss": loss.item(), "Acc": total_correct/total_samples})

        acc = evaluate(model, test_a, DEVICE)
        history["full_curve"].append(acc)
        print(f"Epoch {epoch+1} Test Acc: {acc:.2f}%")

        p1 = model.layer1.P
        print(f"Layer 1 P > 0.5: {(p1>0.5).float().mean()*100:.1f}%")

    print(f"\n[System] Consolidating Memory & Resetting Novices (Percentile {percentile})...")
    static_masks = generate_static_mask_and_reset(model, threshold_percentile=percentile)

    # Phase 2: Train on Task B (digits 5-9) with frozen engram neurons
    print("\n--- Phase 2: Training Task B ---")
    # Reset optimizer for Task B
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

            # Apply gradient masks to protect frozen neurons
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

        # Measure Task A retention
        acc_retention = evaluate(model, test_a, DEVICE)
        history["full_curve"].append(acc_retention)
        print(f"Epoch {epoch+1} Task A Retention: {acc_retention:.2f}%")

        acc_b = evaluate(model, test_b, DEVICE)
        history["task_b"].append(acc_b)
        print(f"Epoch {epoch+1} Task B Accuracy: {acc_b:.2f}%")

    # Final combined evaluation
    acc_all = evaluate(model, test_all, DEVICE)
    print(f"Combined Test Accuracy: {acc_all:.2f}%")

    history["eval_all"] = acc_all
    history["final_task_a"] = history["full_curve"][-1]

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P-Factor No-Scale Ablation Experiment")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs with different seeds")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs per task")
    parser.add_argument("--percentile", type=float, default=0.7, help="Freezing percentile (0.7 = 70%)")
    args = parser.parse_args()

    from src.utils import parse_results_file, save_aggregated_results

    percentile_int = int(args.percentile * 100)
    output_dir = f"results/results_epochs_{args.epochs}"
    os.makedirs(output_dir, exist_ok=True)
    results_file = f"{output_dir}/noscale_{percentile_int}.json"

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
