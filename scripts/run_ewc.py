"""
run_ewc.py - Elastic Weight Consolidation (EWC) Experiment

This script trains an SNN sequentially on Task A then Task B using EWC to mitigate
catastrophic forgetting. EWC computes the Fisher Information Matrix after Task A
and adds a penalty term during Task B to protect important weights.

Usage:
------
    python scripts/run_ewc.py --epochs 3 --runs 5 --ewc_lambda 1000.0
"""

import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import json
import argparse
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.dataset import SpikeMNISTDataset, NMNISTDatasetWrapper
from src.models import SNNModelBaseline

# =============================================================================
# Hyperparameters
# =============================================================================
BATCH_SIZE = 32
HIDDEN_SIZE = 1024
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "spike_mnist_dataset"


def evaluate(model, dataloader, device, task_classes=None):
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
                mask = torch.ones_like(out, dtype=torch.bool)
                mask[:, task_classes] = False
                out[mask] = -float('inf')
                
            preds = out.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
    return total_correct / total_samples * 100 if total_samples > 0 else 0


def compute_fisher(model, dataloader, criterion, num_samples=256, device=DEVICE):
    """Compute empirical Fisher Information Matrix."""
    model.eval()
    fisher = {n: torch.zeros_like(p.data) for n, p in model.named_parameters() if p.requires_grad}
    count = 0
    
    for spikes, labels in dataloader:
        spikes, labels = spikes.to(device), labels.to(device)
        for i in range(spikes.size(0)):
            model.zero_grad()
            model.reset()
            # Batch of size 1
            out = model(spikes[i:i+1])
            loss = criterion(out, labels[i:i+1])
            loss.backward()
            
            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.data ** 2
            
            count += 1
            if count >= num_samples:
                break
        if count >= num_samples:
            break
            
    for n in fisher:
        fisher[n] /= count
    return fisher


def run_experiment(run_id, epochs, seed, ewc_lambda, is_nmnist=False):
    print(f"\n{'='*40}")
    print(f"RUN {run_id+1} (Seed {seed}, EWC Lambda {ewc_lambda})")
    print(f"{'='*40}")
    
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
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    
    if is_nmnist:
        dataset_a = NMNISTDatasetWrapper(save_to=DATA_DIR, train=True, target_digits=[0,1,2,3,4])
        dataset_b = NMNISTDatasetWrapper(save_to=DATA_DIR, train=True, target_digits=[5,6,7,8,9])
        dataset_all = NMNISTDatasetWrapper(save_to=DATA_DIR, train=True, target_digits=list(range(10)))
        input_dim = 2312
    else:
        spike_file = os.path.join(DATA_DIR, "spike_trains_100ts.npy")
        label_file = os.path.join(DATA_DIR, "labels.npy")
        if not os.path.exists(spike_file):
            print("Data not found.")
            return None
        dataset_a = SpikeMNISTDataset(spike_file, label_file, target_digits=[0,1,2,3,4])
        dataset_b = SpikeMNISTDataset(spike_file, label_file, target_digits=[5,6,7,8,9])
        dataset_all = SpikeMNISTDataset(spike_file, label_file, target_digits=list(range(10)))
        input_dim = 784
    
    def get_loader(ds):
        train_size = int(0.8 * len(ds))
        train_ds, test_ds = torch.utils.data.random_split(ds, [train_size, len(ds)-train_size], generator=torch.Generator().manual_seed(seed))
        return DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True), DataLoader(test_ds, batch_size=BATCH_SIZE)

    train_a, test_a = get_loader(dataset_a)
    train_b, test_b = get_loader(dataset_b)
    _, test_all = get_loader(dataset_all)
    
    model = SNNModelBaseline(input_size=input_dim, hidden_size=HIDDEN_SIZE).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    history = {
        "full_curve": [],
        "full_curve_task_il": [],
        "task_b": [],
        "task_b_task_il": [],
        "eval_all": 0.0,
        "final_task_a": 0.0
    }

    # Phase 1: Train on Task A
    print("--- Phase 1: Training Task A ---")
    dataset_name = "NMNIST" if is_nmnist else "MNIST"
    ckpt_dir = os.path.join("checkpoints", dataset_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    start_epoch = 0
    for e in range(epochs, 0, -1):
        temp_ckpt = os.path.join(ckpt_dir, f"baseline_seed_{seed}_epochs_{e}_taskA.pt")
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
                print(f"Error loading checkpoint: {e_msg}")
                continue
            
    if start_epoch > 0:
        acc = evaluate(model, test_a, DEVICE)
        acc_task_il = evaluate(model, test_a, DEVICE, task_classes=[0,1,2,3,4])
        for _ in range(start_epoch):
            history["full_curve"].append(acc)
            history["full_curve_task_il"].append(acc_task_il)
        print(f"Loaded Checkpoint Test Acc (Class-IL): {acc:.2f}% | (Task-IL): {acc_task_il:.2f}%")

    for epoch in range(start_epoch, epochs):
        model.train()
        total_correct = 0
        total_samples = 0
        
        pbar = tqdm(train_a, desc=f"Task A Epoch {epoch+1}")
        for s, l in pbar:
            s, l = s.to(DEVICE), l.to(DEVICE)
            optimizer.zero_grad(); model.reset()
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
        acc_task_il = evaluate(model, test_a, DEVICE, task_classes=[0,1,2,3,4])
        history["full_curve"].append(acc)
        history["full_curve_task_il"].append(acc_task_il)
        print(f"Epoch {epoch+1} Test Acc (Class-IL): {acc:.2f}% | (Task-IL): {acc_task_il:.2f}%")

        temp_ckpt = os.path.join(ckpt_dir, f"baseline_seed_{seed}_epochs_{epoch+1}_taskA.pt")
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'torch_rng_state': torch.get_rng_state(),
            'np_rng_state': np.random.get_state(),
        }
        if torch.cuda.is_available():
            checkpoint['cuda_rng_state'] = torch.cuda.get_rng_state()
        torch.save(checkpoint, temp_ckpt)

    print("\n[System] Computing Fisher Information Matrix (FIM)...")
    fisher = compute_fisher(model, train_a, criterion, num_samples=256, device=DEVICE)
    optimal_params = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}

    # Phase 2: Train on Task B with EWC
    print("\n--- Phase 2: Training Task B (EWC) ---")
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(epochs):
        model.train()
        total_correct = 0
        total_samples = 0
        
        pbar = tqdm(train_b, desc=f"Task B Epoch {epoch+1}")
        for s, l in pbar:
            s, l = s.to(DEVICE), l.to(DEVICE)
            optimizer.zero_grad(); model.reset()
            out = model(s)
            ce_loss = criterion(out, l)
            
            # EWC Penalty
            ewc_loss = 0.0
            for n, p in model.named_parameters():
                if p.requires_grad and n in fisher:
                    ewc_loss += (fisher[n] * (p - optimal_params[n]) ** 2).sum()
                    
            loss = ce_loss + (ewc_lambda * ewc_loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            preds = out.argmax(dim=1)
            total_correct += (preds == l).sum().item()
            total_samples += l.size(0)
            
            pbar.set_postfix({"CE": ce_loss.item(), "EWC": ewc_loss.item(), "Acc": total_correct/total_samples})
            
        acc_retention = evaluate(model, test_a, DEVICE)
        acc_retention_task_il = evaluate(model, test_a, DEVICE, task_classes=[0,1,2,3,4])
        history["full_curve"].append(acc_retention)
        history["full_curve_task_il"].append(acc_retention_task_il)
        print(f"Epoch {epoch+1} Task A Retention (Class-IL): {acc_retention:.2f}% | (Task-IL): {acc_retention_task_il:.2f}%")

        acc_b = evaluate(model, test_b, DEVICE)
        acc_b_task_il = evaluate(model, test_b, DEVICE, task_classes=[5,6,7,8,9])
        history["task_b"].append(acc_b)
        history["task_b_task_il"].append(acc_b_task_il)
        print(f"Epoch {epoch+1} Task B Accuracy (Class-IL): {acc_b:.2f}% | (Task-IL): {acc_b_task_il:.2f}%")
        
    acc_all = evaluate(model, test_all, DEVICE)
    print(f"Combined Test Accuracy: {acc_all:.2f}%")
    
    history["eval_all"] = acc_all
    history["final_task_a"] = history["full_curve"][-1]
    
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EWC Continual Learning Experiment")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs with different seeds")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs per task")
    parser.add_argument("--ewc_lambda", type=float, default=1000.0, help="EWC penalty strength")
    parser.add_argument("--dataset_name", type=str, default="Split-MNIST", help="Dataset name")
    parser.add_argument("--is_nmnist", action="store_true", help="Use NMNIST")
    args = parser.parse_args()
    
    from src.utils import load_legacy_json, parse_results_file, save_aggregated_results
    
    lambda_int = int(args.ewc_lambda)
    results_file = f"results/SNN/{args.dataset_name}/epochs_{args.epochs}/ewc_{lambda_int}.json"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    histories = parse_results_file(results_file)
    existing_seeds = {h['seed'] for h in histories if 'seed' in h}
            
    runs_needed = args.runs - len(histories)
    if runs_needed <= 0:
        print(f"Already have {len(histories)} runs (requested {args.runs}). Skipping...")
        sys.exit(0)
        
    print(f"Found {len(histories)} existing runs. Running {runs_needed} more...")

    runs_completed = 0
    current_seed = 42
    
    while runs_completed < runs_needed:
        if current_seed not in existing_seeds:
            hist = run_experiment(len(histories), args.epochs, current_seed, args.ewc_lambda, args.is_nmnist)
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
