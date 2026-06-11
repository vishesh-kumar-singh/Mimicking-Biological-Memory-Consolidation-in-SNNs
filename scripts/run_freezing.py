"""
run_freezing.py - P-Factor Based Neuron Freezing Experiment

This is the main experiment script for evaluating our proposed method:
using P-factor-based neuron freezing to mitigate catastrophic forgetting.

Experiment Protocol:
-------------------
1. Task A Training: Train on digits 0-4 with LTP/LTD P-factor updates
2. Consolidation: Identify and freeze neurons with high P-factors
3. Task B Training: Train on digits 5-9 with frozen neurons protected
4. Evaluation: Measure retention of Task A and performance on Task B

Key Mechanism:
-------------
- During Task A, neurons that contribute to correct predictions accumulate
  high P-factors via LTP
- At the end of Task A, the top n% of neurons by P-factor are frozen
- Frozen neurons have their gradients masked (set to 0) during Task B
- This preserves the "memory" encoded in high-P neurons while allowing
  plasticity in low-P neurons for learning Task B

This script saves results as JSON files for later analysis.

Usage:
------
    python scripts/run_freezing.py --epochs 3 --percentile 0.7 --runs 5
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
from src.models import SNNModelLTP_LTD, update_p_factor_combined

# =============================================================================
# Hyperparameters
# =============================================================================
BATCH_SIZE = 32       # Mini-batch size for training
HIDDEN_SIZE = 1024    # Number of hidden layer neurons
LR = 1e-3             # Learning rate for Adam optimizer
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "spike_mnist_dataset"


def evaluate(model, dataloader, device, task_classes=None):
    """
    Evaluate model accuracy on a dataloader.
    
    Args:
        model: SNN model to evaluate
        dataloader: DataLoader with test data
        device: torch device
        task_classes: Optional list of valid class indices for Task-IL evaluation
        
    Returns:
        float: Accuracy percentage (0-100)
    """
    model.eval()
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for spikes, labels in dataloader:
            spikes = spikes.to(device)
            labels = labels.to(device)
            model.reset()
            out = model(spikes)
            
            if task_classes is not None:
                # Mask out all classes not belonging to the current task
                mask = torch.ones_like(out, dtype=torch.bool)
                mask[:, task_classes] = False
                out[mask] = -float('inf')
                
            preds = out.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            
    return total_correct / total_samples * 100 if total_samples > 0 else 0


def generate_static_mask_and_reset(model, threshold_percentile=0.7):
    """
    Generate freezing masks based on P-factor values and reset plastic neurons.
    
    This is the core consolidation step between Task A and Task B:
    1. Identify neurons with P-factor in the top (threshold_percentile)
    2. Create gradient masks: 0 for frozen (high P), 1 for plastic (low P)
    3. Re-initialize weights of plastic neurons for Task B learning
    4. Always freeze output heads 0-4 (Task A outputs)
    
    The mask is applied to gradients during Task B training to prevent
    updates to frozen neurons.
    
    Args:
        model (SNNModelLTP_LTD): Trained model with P-factors
        threshold_percentile (float): Fraction of neurons to freeze (0.7 = 70%)
        
    Returns:
        dict: Gradient masks for each layer
            - 'layer1': [hidden_size, 1] mask
            - 'layer2': [output_size, 1] mask
    """
    masks = {}
    
    if hasattr(model, 'layer1') and hasattr(model.layer1, 'P'):
        p1 = model.layer1.P.detach()
        
        # Calculate the P-value threshold for freezing
        # k = number of neurons to freeze
        k = int(p1.size(0) * threshold_percentile)
        # Find the k-th largest P value (neurons with P >= threshold are frozen)
        threshold1 = torch.kthvalue(p1, p1.size(0) - k + 1).values
        
        # Create mask: 0 for frozen (high P), 1 for plastic (low P)
        mask1 = (p1 < threshold1).float().unsqueeze(1).to(p1.device)
        masks['layer1'] = mask1
        
        # Reset weights of plastic neurons (the "novice" neurons that will learn Task B)
        novice_indices = torch.where(mask1.squeeze() == 1)[0]
        if len(novice_indices) > 0:
            with torch.no_grad():
                nn.init.kaiming_uniform_(model.layer1.linear.weight[novice_indices], a=np.sqrt(5))
                if model.layer1.linear.bias is not None:
                     nn.init.zeros_(model.layer1.linear.bias[novice_indices])

    if hasattr(model, 'layer2'):
        # Output layer: always freeze Task A outputs (neurons 0-4)
        # This prevents Task A predictions from being corrupted
        mask2 = torch.ones(model.layer2.linear.weight.shape[0], 1).to(p1.device)
        mask2[0:5] = 0.0  # Freeze outputs 0-4 (Task A digits)
        masks['layer2'] = mask2
        
        # Reset Task B output heads (neurons 5-9)
        if hasattr(model.layer2, 'P'):
            with torch.no_grad():
                model.layer2.P[5:] = 0.0  # Reset P-factors for Task B outputs
                nn.init.kaiming_uniform_(model.layer2.linear.weight[5:], a=np.sqrt(5))
                if model.layer2.linear.bias is not None:
                    nn.init.zeros_(model.layer2.linear.bias[5:])
                    
    return masks


def run_experiment(run_id, epochs, seed, percentile, data_dir=DATA_DIR, is_nmnist=False, alpha_ltp=0.01, alpha_ltd=0.01):
    """
    Run a single continual learning experiment with P-factor freezing.
    
    Args:
        run_id (int): Experiment run identifier
        epochs (int): Training epochs per task
        seed (int): Random seed for reproducibility
        percentile (float): Fraction of neurons to freeze
        data_dir (str): Directory containing dataset
        is_nmnist (bool): Whether to use NMNISTDatasetWrapper
        alpha_ltp (float): LTP learning rate
        alpha_ltd (float): LTD learning rate
        
    Returns:
        dict: Experiment results containing:
            - full_curve: Task A accuracy after each epoch
            - task_b: Task B accuracy after each epoch
            - eval_all: Combined accuracy on all digits
            - final_task_a: Final Task A retention
    """
    print(f"\n{'='*40}")
    print(f"RUN {run_id+1} (Seed {seed}, Percentile {percentile})")
    print(f"{'='*40}")
    
    import random
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    
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
        """Split dataset 80/20 and create DataLoaders."""
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
    
    # Initialize LTP/LTD SNN model
    model = SNNModelLTP_LTD(input_size=input_dim, hidden_size=HIDDEN_SIZE).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    history = {
        "full_curve": [],         # Task A accuracy over all epochs (Class-IL)
        "full_curve_task_il": [], # Task A accuracy over all epochs (Task-IL)
        "task_b": [],             # Task B accuracy during Phase 2 (Class-IL)
        "task_b_task_il": [],     # Task B accuracy during Phase 2 (Task-IL)
        "eval_all": 0.0,          # Final combined accuracy
        "final_task_a": 0.0       # Final Task A retention
    }

    # =========================================================================
    # PHASE 1: Train on Task A (digits 0-4) with LTP/LTD
    # =========================================================================
    print("--- Phase 1: Training Task A ---")
    
    # Checkpoint path for Task A
    dataset_name = "NMNIST" if is_nmnist else "MNIST"
    ckpt_dir = os.path.join("checkpoints", dataset_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"seed_{seed}_epochs_{epochs}_ltp{alpha_ltp}_ltd{alpha_ltd}_taskA.pt")
    
    if os.path.exists(ckpt_path):
        print(f"Loading Task A state from checkpoint: {ckpt_path}")
        try:
            checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            torch.set_rng_state(checkpoint['torch_rng_state'].cpu())
            np.random.set_state(checkpoint['np_rng_state'])
            if torch.cuda.is_available() and 'cuda_rng_state' in checkpoint:
                torch.cuda.set_rng_state(checkpoint['cuda_rng_state'].cpu())
            start_epoch = epochs
        except Exception as e_msg:
            print(f"Error loading full state checkpoint: {e_msg}. Ignoring.")
            start_epoch = 0
    else:
        start_epoch = 0
        for e in range(epochs, 0, -1):
            temp_ckpt = os.path.join(ckpt_dir, f"seed_{seed}_epochs_{e}_ltp{alpha_ltp}_ltd{alpha_ltd}_taskA.pt")
            if os.path.exists(temp_ckpt):
                print(f"Loading Task A state from checkpoint: {temp_ckpt}")
                try:
                    checkpoint = torch.load(temp_ckpt, map_location=DEVICE, weights_only=False)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    torch.set_rng_state(checkpoint['torch_rng_state'].cpu())
                    np.random.set_state(checkpoint['np_rng_state'])
                    if torch.cuda.is_available() and 'cuda_rng_state' in checkpoint:
                        torch.cuda.set_rng_state(checkpoint['cuda_rng_state'].cpu())
                    start_epoch = e
                    break
                except Exception as e_msg:
                    print(f"Error loading full state checkpoint: {e_msg}. Ignoring.")
                    continue

    if start_epoch > 0:
        # We still need to evaluate to populate history
        acc = evaluate(model, test_a, DEVICE)
        acc_task_il = evaluate(model, test_a, DEVICE, task_classes=[0,1,2,3,4])
        for _ in range(start_epoch):
            history["full_curve"].append(acc)
            history["full_curve_task_il"].append(acc_task_il)
            
        print(f"Loaded Checkpoint Test Acc (Class-IL): {acc:.2f}% | (Task-IL): {acc_task_il:.2f}%")
        p1 = model.layer1.P
        print(f"Layer 1 P > 0.5: {(p1>0.5).float().mean()*100:.1f}%")
        
    for epoch in range(start_epoch, epochs):
            model.train()
            total_correct = 0
            total_samples = 0
            
            pbar = tqdm(train_a, desc=f"Task A Epoch {epoch+1}")
            for s, l in pbar:
                s, l = s.to(DEVICE), l.to(DEVICE)
                
                # Forward pass
                optimizer.zero_grad()
                model.reset()
                out = model(s)
                loss = criterion(out, l)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Track accuracy
                preds = out.argmax(dim=1)
                total_correct += (preds == l).sum().item()
                total_samples += l.size(0)
                
                # Apply LTP/LTD P-factor updates based on prediction correctness
                update_p_factor_combined(model, preds, l, alpha_ltp=alpha_ltp, alpha_ltd=alpha_ltd)
                
                pbar.set_postfix({"Loss": loss.item(), "Acc": total_correct/total_samples})
                    
            # Evaluate Task A performance
            acc = evaluate(model, test_a, DEVICE)
            acc_task_il = evaluate(model, test_a, DEVICE, task_classes=[0,1,2,3,4])
            history["full_curve"].append(acc)
            history["full_curve_task_il"].append(acc_task_il)
            print(f"Epoch {epoch+1} Test Acc (Class-IL): {acc:.2f}% | (Task-IL): {acc_task_il:.2f}%")
            
            # Report P-factor statistics
            p1 = model.layer1.P
            print(f"Layer 1 P > 0.5: {(p1>0.5).float().mean()*100:.1f}%")
    
            # SAVE EVERY EPOCH WITH FULL STATE
            temp_ckpt = os.path.join(ckpt_dir, f"seed_{seed}_epochs_{epoch+1}_ltp{alpha_ltp}_ltd{alpha_ltd}_taskA.pt")
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'torch_rng_state': torch.get_rng_state(),
                'np_rng_state': np.random.get_state(),
            }
            if torch.cuda.is_available():
                checkpoint['cuda_rng_state'] = torch.cuda.get_rng_state()
            torch.save(checkpoint, temp_ckpt)

    # =========================================================================
    # CONSOLIDATION: Freeze high-P neurons, reset low-P neurons
    # =========================================================================
    print(f"\n[System] Consolidating Memory & Resetting Novices (Percentile {percentile})...")
    static_masks = generate_static_mask_and_reset(model, threshold_percentile=percentile)

    # =========================================================================
    # PHASE 2: Train on Task B (digits 5-9) with frozen neurons
    # =========================================================================
    print("\n--- Phase 2: Training Task B ---")
    # Re-initialize optimizer (clear momentum from Task A)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(epochs):
        model.train()
        total_correct = 0
        total_samples = 0
        
        pbar = tqdm(train_b, desc=f"Task B Epoch {epoch+1}")
        for s, l in pbar:
            s, l = s.to(DEVICE), l.to(DEVICE)
            
            # Forward pass
            optimizer.zero_grad()
            model.reset()
            out = model(s)
            loss = criterion(out, l)
            
            # Backward pass with gradient masking
            loss.backward()
            
            # Apply freezing masks to gradients
            if static_masks:
                if model.layer1.linear.weight.grad is not None:
                    model.layer1.linear.weight.grad.data.mul_(static_masks['layer1'])
                if model.layer2.linear.weight.grad is not None:
                    model.layer2.linear.weight.grad.data.mul_(static_masks['layer2'])
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Track accuracy
            preds = out.argmax(dim=1)
            total_correct += (preds == l).sum().item()
            total_samples += l.size(0)
            
            # Apply LTP/LTD P-factor updates based on prediction correctness during Task B as well
            update_p_factor_combined(model, preds, l, alpha_ltp=alpha_ltp, alpha_ltd=alpha_ltd, static_masks=static_masks)
            
            pbar.set_postfix({"Loss": loss.item(), "Acc": total_correct/total_samples})
            
        # Evaluate Task A retention (catastrophic forgetting metric)
        acc_retention = evaluate(model, test_a, DEVICE)
        acc_retention_task_il = evaluate(model, test_a, DEVICE, task_classes=[0,1,2,3,4])
        history["full_curve"].append(acc_retention)
        history["full_curve_task_il"].append(acc_retention_task_il)
        print(f"Epoch {epoch+1} Task A Retention (Class-IL): {acc_retention:.2f}% | (Task-IL): {acc_retention_task_il:.2f}%")

        # Evaluate Task B learning
        acc_b = evaluate(model, test_b, DEVICE)
        acc_b_task_il = evaluate(model, test_b, DEVICE, task_classes=[5,6,7,8,9])
        history["task_b"].append(acc_b)
        history["task_b_task_il"].append(acc_b_task_il)
        print(f"Epoch {epoch+1} Task B Accuracy (Class-IL): {acc_b:.2f}% | (Task-IL): {acc_b_task_il:.2f}%")
        
    # =========================================================================
    # FINAL EVALUATION
    # =========================================================================
    acc_all = evaluate(model, test_all, DEVICE)
    print(f"Combined Test Accuracy: {acc_all:.2f}%")
    
    history["eval_all"] = acc_all
    history["final_task_a"] = history["full_curve"][-1]
    
    return history


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P-Factor Freezing Experiment")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs with different seeds")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs per task")
    parser.add_argument("--percentile", type=float, default=0.7, help="Freezing percentile (0.7 = 70%)")
    parser.add_argument("--data_dir", type=str, default="spike_mnist_dataset", help="Directory with spike data")
    parser.add_argument("--dataset_name", type=str, default="Split-MNIST", help="Name for results directory")
    parser.add_argument("--is_nmnist", action="store_true", help="Use NMNISTDatasetWrapper")
    parser.add_argument("--alpha_ltp", type=float, default=0.01, help="LTP learning rate")
    parser.add_argument("--alpha_ltd", type=float, default=0.01, help="LTD learning rate")
    args = parser.parse_args()
    
    from src.utils import load_legacy_json, parse_results_file, save_aggregated_results
    
    # Setup output path
    percentile_int = int(args.percentile * 100)
    output_dir = f"results/SNN/{args.dataset_name}/epochs_{args.epochs}"
    os.makedirs(output_dir, exist_ok=True)
    results_file = f"{output_dir}/freezing_{percentile_int}.json"
    
    # Load existing results (supports resuming interrupted experiments)
    histories = parse_results_file(results_file)
    
    # Legacy support: migrate old-format results if needed
    if not histories and args.percentile == 0.7:
        print("No aggregated results found. Checking for legacy JSON files...")
        legacy_histories = load_legacy_json("results", "freezing_run_*.json")
        if legacy_histories:
            print(f"Found {len(legacy_histories)} legacy runs. Merging...")
            histories.extend(legacy_histories)
    
    # Determine how many more runs are needed
    existing_seeds = set()
    for h in histories:
        if 'seed' in h:
            existing_seeds.add(h['seed'])
            
    runs_needed = args.runs - len(histories)
    if runs_needed <= 0:
        print(f"Already have {len(histories)} runs (requested {args.runs}). Skipping...")
        sys.exit(0)
        
    print(f"Found {len(histories)} existing runs. Running {runs_needed} more...")

    # Run experiments with different seeds
    runs_completed = 0
    current_seed = 42  # Starting seed
    
    while runs_completed < runs_needed:
        if current_seed not in existing_seeds:
            hist = run_experiment(len(histories), args.epochs, current_seed, args.percentile, data_dir=args.data_dir, is_nmnist=args.is_nmnist, alpha_ltp=args.alpha_ltp, alpha_ltd=args.alpha_ltd)
            if hist:
                hist['seed'] = current_seed
                histories.append(hist)
                # Save after each run (checkpoint)
                save_aggregated_results(results_file, histories)
                print(f"Saved result for seed {current_seed} to {results_file}")
            runs_completed += 1
        current_seed += 1
    
    # Handle edge case: just re-aggregate existing results
    if args.runs == 0 and histories:
         save_aggregated_results(results_file, histories)
         print(f"Aggregated results saved to {results_file}")
