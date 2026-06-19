"""
run_fashion_mnist_sweep.py - Fashion-MNIST Hyperparameter Sweep for P-Factor Freezing

This script is SEPARATE from the Split-MNIST pipeline. It sweeps hyperparameters
to find optimal P-Factor configurations for 5-Split Fashion-MNIST.

Key differences from run_multitask_unified.py:
- Configurable alpha_ltp and alpha_ltd (the main tuning knobs)
- Configurable hidden_size (supports 2048+)
- Configurable learning rate
- Runs a grid search over multiple configs
- Saves results with config-specific filenames

Usage:
    # Run a single config:
    python experiments/run_fashion_mnist_sweep.py --mode single --percentile 0.15 \
        --alpha_ltp 0.05 --alpha_ltd 0.005 --hidden_size 1024 --epochs 2 --runs 2

    # Run the full sweep:
    python experiments/run_fashion_mnist_sweep.py --mode sweep --runs 2 --epochs 2
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
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.dataset import SpikeMNISTDataset
from src.models import SNNModelLTP_LTD, update_p_factor_combined, SNNModelBaseline

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "spike_fashion_mnist_dataset"
BATCH_SIZE = 32


def get_task_class_splits(num_tasks):
    """Return list of class lists for each task.
    5-split: [[0,1],[2,3],[4,5],[6,7],[8,9]]
    4-split: [[0,1],[2,3,4],[5,6],[7,8,9]]  (2+3+2+3=10)
    3-split: [[0,1,2],[3,4,5],[6,7,8,9]]    (3+3+4=10)
    """
    if num_tasks == 5:
        return [[0,1],[2,3],[4,5],[6,7],[8,9]]
    elif num_tasks == 4:
        return [[0,1],[2,3,4],[5,6],[7,8,9]]
    elif num_tasks == 3:
        return [[0,1,2],[3,4,5],[6,7,8,9]]
    else:
        raise ValueError(f"Unsupported num_tasks={num_tasks}")


def evaluate_tasks(model, dataloaders, device, max_task_idx, task_splits):
    """Evaluate model on all tasks up to max_task_idx for both Class-IL and Task-IL."""
    model.eval()
    accs_class_il = []
    accs_task_il = []

    with torch.no_grad():
        for task_idx in range(max_task_idx + 1):
            total_correct_class = 0
            total_correct_task = 0
            total_samples = 0
            loader = dataloaders[task_idx]

            task_classes = task_splits[task_idx]

            for spikes, labels in loader:
                spikes, labels = spikes.to(device), labels.to(device)
                model.reset()
                out = model(spikes)

                # Class-IL Prediction (argmax over all 10 classes)
                preds_class = out.argmax(dim=1)
                total_correct_class += (preds_class == labels).sum().item()

                # Task-IL Prediction (argmax over just the classes of this task)
                out_task = out.clone()
                mask = torch.ones_like(out_task, dtype=torch.bool)
                mask[:, task_classes] = False
                out_task[mask] = -float('inf')
                preds_task = out_task.argmax(dim=1)
                total_correct_task += (preds_task == labels).sum().item()

                total_samples += labels.size(0)

            acc_class = total_correct_class / total_samples * 100 if total_samples > 0 else 0
            acc_task = total_correct_task / total_samples * 100 if total_samples > 0 else 0
            accs_class_il.append(acc_class)
            accs_task_il.append(acc_task)

    return accs_class_il, accs_task_il


def generate_static_mask(model, frozen_mask, task_idx, percentile=0.2, hidden_size=1024, task_splits=None):
    """
    Generate freezing masks based on P-factor values.
    No-reset variant (matching the best Split-MNIST config).
    """
    masks = {}

    if hasattr(model, 'layer1'):
        plastic_indices = torch.where(~frozen_mask)[0]

        if len(plastic_indices) > 0:
            k = min(int(hidden_size * percentile), len(plastic_indices))
            newly_frozen = torch.zeros_like(frozen_mask)

            if k > 0 and hasattr(model.layer1, 'P'):
                p1 = model.layer1.P.detach()
                p1_plastic = p1[plastic_indices]
                threshold = torch.kthvalue(p1_plastic, len(plastic_indices) - k + 1).values
                newly_frozen = (p1 >= threshold) & (~frozen_mask)

                frozen_mask = frozen_mask | newly_frozen

        masks['layer1'] = (~frozen_mask).float().unsqueeze(1).to(DEVICE)

    if hasattr(model, 'layer2'):
        # Freeze output heads for all classes seen so far
        if task_splits is not None:
            seen_classes = []
            for t in range(task_idx + 1):
                seen_classes.extend(task_splits[t])
        else:
            seen_classes = list(range((task_idx + 1) * 2))
        mask2 = torch.ones(model.layer2.linear.weight.shape[0], 1).to(DEVICE)
        for c in seen_classes:
            mask2[c] = 0.0
        masks['layer2'] = mask2

    return masks, frozen_mask

def generate_random_mask(model, frozen_mask, task_idx, percentile=0.15, hidden_size=1024, task_splits=None):
    """
    Generate freezing masks based on RANDOM selection.
    """
    masks = {}
    if hasattr(model, 'layer1'):
        plastic_indices = torch.where(~frozen_mask)[0]
        if len(plastic_indices) > 0:
            k = min(int(hidden_size * percentile), len(plastic_indices))
            newly_frozen = torch.zeros_like(frozen_mask)
            if k > 0:
                perm = torch.randperm(len(plastic_indices))
                newly_frozen_idx = plastic_indices[perm[:k]]
                newly_frozen[newly_frozen_idx] = True
                frozen_mask = frozen_mask | newly_frozen

        masks['layer1'] = (~frozen_mask).float().unsqueeze(1).to(DEVICE)

    if hasattr(model, 'layer2'):
        if task_splits is not None:
            seen_classes = []
            for t in range(task_idx + 1):
                seen_classes.extend(task_splits[t])
        else:
            seen_classes = list(range((task_idx + 1) * 2))
        mask2 = torch.ones(model.layer2.linear.weight.shape[0], 1).to(DEVICE)
        for c in seen_classes:
            mask2[c] = 0.0
        masks['layer2'] = mask2

    return masks, frozen_mask

def generate_packnet_mask(model, frozen_weight_mask, task_idx, percentile=0.15, task_splits=None):
    """
    Generate PackNet weight masks (per weight magnitude).
    """
    masks = {}
    if hasattr(model, 'layer1'):
        W_abs = torch.abs(model.layer1.linear.weight.data)
        plastic_mask = ~frozen_weight_mask
        
        # We need to freeze `percentile` fraction of the TOTAL weights.
        # But only from the currently plastic weights.
        total_weights = W_abs.numel()
        k = int(total_weights * percentile)
        
        plastic_weights = W_abs[plastic_mask]
        if len(plastic_weights) > 0:
            k = min(k, len(plastic_weights))
            if k > 0:
                threshold = torch.kthvalue(plastic_weights, len(plastic_weights) - k + 1).values
                newly_frozen = (W_abs >= threshold) & plastic_mask
                frozen_weight_mask = frozen_weight_mask | newly_frozen
                
                # Prune the REMAINING plastic weights to zero
                still_plastic = ~frozen_weight_mask
                model.layer1.linear.weight.data *= (~still_plastic).float()
                
        masks['layer1'] = (~frozen_weight_mask).float()

    if hasattr(model, 'layer2'):
        if task_splits is not None:
            seen_classes = []
            for t in range(task_idx + 1):
                seen_classes.extend(task_splits[t])
        else:
            seen_classes = list(range((task_idx + 1) * 2))
        mask2 = torch.ones(model.layer2.linear.weight.shape[0], 1).to(DEVICE)
        for c in seen_classes:
            mask2[c] = 0.0
        masks['layer2'] = mask2
        
        with torch.no_grad():
            for c in range(10):
                if c not in seen_classes:
                    model.layer2.linear.weight.data[c] = 0.0

    return masks, frozen_weight_mask

def compute_fisher(model, dataloader, criterion, device, seen_classes_sorted, label_remap, num_samples=256):
    """Compute empirical Fisher Information Matrix for EWC."""
    model.eval()
    fisher = {n: torch.zeros_like(p.data) for n, p in model.named_parameters() if p.requires_grad}
    count = 0
    
    for spikes, labels in dataloader:
        spikes, labels = spikes.to(device), labels.to(device)
        for i in range(spikes.size(0)):
            model.zero_grad()
            model.reset()
            out = model(spikes[i:i+1])
            out_seen = out[:, seen_classes_sorted]
            l_local = labels[i:i+1].clone()
            for orig_c, local_idx in label_remap.items():
                l_local[l_local == orig_c] = local_idx
            loss = criterion(out_seen, l_local)
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
        fisher[n] /= max(1, count)
    return fisher

def run_experiment(run_id, epochs, seed, percentile, hidden_size, alpha_ltp, alpha_ltd, lr, num_tasks=5, er_mode=False, er_buffer=200, baseline_mode=False, ewc_mode=False, ewc_lambda=1000.0, si_mode=False, si_c=1.0, si_xi=0.1, random_mode=False, packnet_mode=False):
    """Run a single N-task continual learning experiment."""
    if baseline_mode: mode_str = "Fine-Tuning Baseline"; percentile = 0.0
    elif er_mode: mode_str = "ER Baseline"; percentile = 0.0
    elif ewc_mode: mode_str = f"EWC (lambda={ewc_lambda})"; percentile = 0.0
    elif si_mode: mode_str = f"SI (c={si_c})"; percentile = 0.0
    elif random_mode: mode_str = "Random Freezing"
    elif packnet_mode: mode_str = "PackNet"
    else: mode_str = "P-Factor"
    
    print(f"\n{'='*40}")
    print(f"RUN {run_id+1} ({mode_str} | Seed {seed}, p={percentile}, h={hidden_size}, "
          f"ltp={alpha_ltp}, ltd={alpha_ltd}, lr={lr}, tasks={num_tasks})")
    print(f"{'='*40}")

    task_splits = get_task_class_splits(num_tasks)

    # Seed everything
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

    # Load data
    spike_file = os.path.join(DATA_DIR, "spike_trains_100ts.npy")
    label_file = os.path.join(DATA_DIR, "labels.npy")
    if not os.path.exists(spike_file):
        print(f"Data not found in {DATA_DIR}.")
        return None

    train_loaders = []
    test_loaders = []

    for t in range(num_tasks):
        target_digits = task_splits[t]
        ds = SpikeMNISTDataset(spike_file, label_file, target_digits=target_digits)
        train_size = int(0.8 * len(ds))
        train_ds, test_ds = torch.utils.data.random_split(
            ds, [train_size, len(ds) - train_size],
            generator=torch.Generator().manual_seed(seed)
        )
        train_loaders.append(DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True))
        test_loaders.append(DataLoader(test_ds, batch_size=BATCH_SIZE))

    # Initialize model
    if ewc_mode or si_mode or baseline_mode or er_mode or random_mode or packnet_mode:
        model = SNNModelBaseline(input_size=784, hidden_size=hidden_size).to(DEVICE)
    else:
        model = SNNModelLTP_LTD(input_size=784, hidden_size=hidden_size).to(DEVICE)
        
    criterion = nn.CrossEntropyLoss()
    acc_matrix_class_il = []
    acc_matrix_task_il = []

    # State tracking
    frozen_mask = torch.zeros(hidden_size, dtype=torch.bool).to(DEVICE)
    static_masks = None
    
    # EWC/SI tracking
    fisher_matrix = {n: torch.zeros_like(p.data) for n, p in model.named_parameters() if p.requires_grad}
    optimal_params = {}
    omega = {n: torch.zeros_like(p.data) for n, p in model.named_parameters() if p.requires_grad}
    Omega = {}
    p_old = {}
    
    # PackNet tracking
    frozen_weight_mask = torch.zeros((hidden_size, 784), dtype=torch.bool).to(DEVICE)
    
    # ER tracking
    buffer_x = []
    buffer_y = []
    buffer_x_tensor = None
    buffer_y_tensor = None

    for task_idx in range(num_tasks):
        classes_str = ','.join(map(str, task_splits[task_idx]))
        print(f"\n--- Training Task {task_idx+1}/{num_tasks} (Classes {classes_str}) ---")

        seen_classes = []
        for t in range(task_idx + 1):
            seen_classes.extend(task_splits[t])
        seen_classes_sorted = sorted(seen_classes)
        label_remap = {c: i for i, c in enumerate(seen_classes_sorted)}

        if hasattr(model, 'layer2'):
            with torch.no_grad():
                for c in task_splits[task_idx]:
                    if hasattr(model.layer2, 'P'):
                        model.layer2.P[c] = 0.0
                    import math
                    nn.init.kaiming_uniform_(model.layer2.linear.weight[c:c+1], a=math.sqrt(5))
                    if model.layer2.linear.bias is not None:
                        nn.init.zeros_(model.layer2.linear.bias[c:c+1])

        train_loader = train_loaders[task_idx]
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(epochs):
            model.train()
            total_correct = 0
            total_samples = 0

            pbar = tqdm(train_loader, desc=f"T{task_idx+1} Epoch {epoch+1}")
            for s, l in pbar:
                s, l = s.to(DEVICE), l.to(DEVICE)

                if er_mode and task_idx > 0 and len(buffer_y) > 0:
                    bz = min(BATCH_SIZE // 2, len(buffer_y))
                    idx = torch.randperm(len(buffer_y))[:bz]
                    rx = buffer_x_tensor[idx].to(DEVICE)
                    ry = buffer_y_tensor[idx].to(DEVICE)
                    s = torch.cat([s, rx], dim=0)
                    l = torch.cat([l, ry], dim=0)
                    perm = torch.randperm(len(l))
                    s = s[perm]
                    l = l[perm]

                optimizer.zero_grad()
                model.reset()
                
                # For SI: save old weights
                if si_mode:
                    for n, p in model.named_parameters():
                        if p.requires_grad:
                            p_old[n] = p.data.clone()

                out = model(s)
                
                out_seen = out[:, seen_classes_sorted]
                l_local = l.clone()
                for orig_c, local_idx in label_remap.items():
                    l_local[l == orig_c] = local_idx
                ce_loss = criterion(out_seen, l_local)
                loss = ce_loss

                # EWC Penalty
                ewc_loss = 0.0
                if ewc_mode and task_idx > 0:
                    for n, p in model.named_parameters():
                        if p.requires_grad and n in fisher_matrix:
                            ewc_loss += (fisher_matrix[n] * (p - optimal_params[n]) ** 2).sum()
                    loss = ce_loss + (ewc_lambda * ewc_loss)
                    
                # SI Penalty
                si_loss = 0.0
                if si_mode and task_idx > 0:
                    for n, p in model.named_parameters():
                        if p.requires_grad and n in Omega:
                            si_loss += (Omega[n] * (p - optimal_params[n]) ** 2).sum()
                    loss = ce_loss + (si_c * si_loss)

                loss.backward()

                # Apply freezing masks
                if static_masks:
                    if model.layer1.linear.weight.grad is not None:
                        model.layer1.linear.weight.grad.data.mul_(static_masks['layer1'])
                    if model.layer2.linear.weight.grad is not None:
                        model.layer2.linear.weight.grad.data.mul_(static_masks['layer2'])

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # SI Path Integral Update
                if si_mode:
                    for n, p in model.named_parameters():
                        if p.requires_grad and p.grad is not None:
                            delta_w = p.data - p_old[n]
                            omega[n] += -p.grad.data * delta_w

                out_for_preds = out.clone()
                future_classes = [c for c in range(10) if c not in seen_classes_sorted]
                if future_classes:
                    out_for_preds[:, future_classes] = -float('inf')
                
                preds = out_for_preds.argmax(dim=1)
                total_correct += (preds == l).sum().item()
                total_samples += l.size(0)

                if not (er_mode or baseline_mode or ewc_mode or si_mode or random_mode or packnet_mode):
                    update_p_factor_combined(
                        model, preds, l,
                        alpha_ltp=alpha_ltp, alpha_ltd=alpha_ltd,
                        static_masks=static_masks
                    )

                postfix = {"Loss": loss.item(), "Acc": total_correct / total_samples}
                if ewc_mode and task_idx > 0: postfix["EWC"] = ewc_loss.item()
                if si_mode and task_idx > 0: postfix["SI"] = si_loss.item()
                pbar.set_postfix(postfix)

        if task_idx < num_tasks - 1:
            if er_mode:
                print(f"[System] Populating ER Buffer with Task {task_idx+1} data...")
                loader = train_loaders[task_idx]
                counts = {c: 0 for c in task_splits[task_idx]}
                samples_per_class = er_buffer
                for spikes, labels in loader:
                    for i in range(len(labels)):
                        c = labels[i].item()
                        if counts[c] < samples_per_class:
                            buffer_x.append(spikes[i:i+1].cpu())
                            buffer_y.append(labels[i:i+1].cpu())
                            counts[c] += 1
                    if all(counts[c] >= samples_per_class for c in task_splits[task_idx]):
                        break
                buffer_x_tensor = torch.cat(buffer_x, dim=0)
                buffer_y_tensor = torch.cat(buffer_y, dim=0)
                print(f"Total ER Buffer size: {len(buffer_y_tensor)}")
                
            elif ewc_mode:
                print(f"[System] Computing Fisher Information Matrix for Task {task_idx+1}...")
                task_fisher = compute_fisher(model, train_loaders[task_idx], criterion, DEVICE, seen_classes_sorted, label_remap)
                for n in fisher_matrix:
                    fisher_matrix[n] += task_fisher[n]
                optimal_params = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
                
            elif si_mode:
                print(f"[System] Computing SI Importance Matrix for Task {task_idx+1}...")
                for n, p in model.named_parameters():
                    if p.requires_grad:
                        optimal_params[n] = p.data.clone()
                        Omega[n] = torch.relu(omega[n]) / (torch.abs(p.data) + si_xi)
                # We do not reset omega in SI across tasks (usually it's a running sum of paths)
                
            elif random_mode:
                print(f"[System] Consolidating Memory (Random)...")
                static_masks, frozen_mask = generate_random_mask(
                    model, frozen_mask, task_idx,
                    percentile=percentile, hidden_size=hidden_size,
                    task_splits=task_splits
                )
                frozen_count = frozen_mask.sum().item()
                print(f"Total neurons frozen: {frozen_count} / {hidden_size}")
                
            elif packnet_mode:
                print(f"[System] Consolidating Memory (PackNet)...")
                static_masks, frozen_weight_mask = generate_packnet_mask(
                    model, frozen_weight_mask, task_idx,
                    percentile=percentile, task_splits=task_splits
                )
                frozen_count = frozen_weight_mask.sum().item()
                total_weights = hidden_size * 784
                print(f"Total weights frozen: {frozen_count} / {total_weights}")
                
            elif not baseline_mode:
                print(f"[System] Consolidating Memory (P-Factor)...")
                static_masks, frozen_mask = generate_static_mask(
                    model, frozen_mask, task_idx,
                    percentile=percentile, hidden_size=hidden_size,
                    task_splits=task_splits
                )
                frozen_count = frozen_mask.sum().item()
                print(f"Total neurons frozen: {frozen_count} / {hidden_size}")

        accs_class, accs_task = evaluate_tasks(model, test_loaders, DEVICE, task_idx, task_splits)
        acc_matrix_class_il.append(accs_class)
        acc_matrix_task_il.append(accs_task)

        print(f"Task {task_idx+1} completed. Accuracies on seen tasks:")
        for i, (ac, at) in enumerate(zip(accs_class, accs_task)):
            print(f"  -> Task {i+1}: Class-IL: {ac:.2f}% | Task-IL: {at:.2f}%")

    final_avg = sum(acc_matrix_class_il[-1]) / len(acc_matrix_class_il[-1])
    final_avg_task = sum(acc_matrix_task_il[-1]) / len(acc_matrix_task_il[-1])

    print(f"\n*** FINAL: Class-IL Avg={final_avg:.2f}%, Task-IL Avg={final_avg_task:.2f}% ***")

    history = {
        "acc_matrix": acc_matrix_class_il,
        "acc_matrix_class_il": acc_matrix_class_il,
        "acc_matrix_task_il": acc_matrix_task_il,
        "final_avg_acc": final_avg,
        "final_avg_acc_task_il": final_avg_task,
        "eval_all": final_avg,
        "final_task_a": acc_matrix_class_il[-1][0] if acc_matrix_class_il else 0
    }
    return history


def run_config(percentile, alpha_ltp, alpha_ltd, hidden_size, epochs, lr, num_runs, seed_start=42, num_tasks=5, 
               er_mode=False, er_buffer=200, baseline_mode=False, ewc_mode=False, ewc_lambda=1000.0,
               si_mode=False, si_c=1.0, si_xi=0.1, random_mode=False, packnet_mode=False, output_dir=None):
    """Run a specific config for num_runs seeds and return results."""
    if baseline_mode:
        config_name = "cl_baseline"
    elif er_mode:
        config_name = f"fmnist_{num_tasks}t_er{er_buffer}_h{hidden_size}_e{epochs}"
    elif ewc_mode:
        config_name = f"fmnist_{num_tasks}t_ewc_{int(ewc_lambda)}_e{epochs}"
    elif si_mode:
        config_name = f"fmnist_{num_tasks}t_si_{int(si_c)}_e{epochs}"
    elif random_mode:
        config_name = f"fmnist_{num_tasks}t_random_p{int(percentile*100)}_e{epochs}"
    elif packnet_mode:
        config_name = f"fmnist_{num_tasks}t_packnet_p{int(percentile*100)}_e{epochs}"
    else:
        config_name = f"fmnist_{num_tasks}t_p{int(percentile*100)}_h{hidden_size}_ltp{alpha_ltp}_ltd{alpha_ltd}_e{epochs}_lr{lr}"
        
    print(f"\n{'#'*60}")
    print(f"CONFIG: {config_name}")
    print(f"{'#'*60}")

    histories = []
    if output_dir:
        from src.utils import parse_results_file
        results_file = os.path.join(output_dir, f"{config_name}.json")
        if os.path.exists(results_file):
            histories = parse_results_file(results_file)
            print(f"Found {len(histories)} existing runs for {config_name}.")
            
    existing_seeds = {h['seed'] for h in histories if 'seed' in h}
    runs_needed = num_runs - len(histories)

    if runs_needed <= 0:
        print(f"Already have {len(histories)} runs. Skipping.")
    else:
        runs_completed = 0
        current_seed = seed_start
        
        while runs_completed < runs_needed:
            if current_seed not in existing_seeds:
                hist = run_experiment(
                    len(histories), epochs, current_seed, percentile, hidden_size, alpha_ltp, alpha_ltd, lr, 
                    num_tasks=num_tasks, er_mode=er_mode, er_buffer=er_buffer, baseline_mode=baseline_mode,
                    ewc_mode=ewc_mode, ewc_lambda=ewc_lambda, si_mode=si_mode, si_c=si_c, si_xi=si_xi,
                    random_mode=random_mode, packnet_mode=packnet_mode
                )
                if hist:
                    hist['seed'] = current_seed
                    if not (er_mode or ewc_mode or si_mode or baseline_mode):
                        hist['percentile'] = percentile
                    if not (er_mode or ewc_mode or si_mode or baseline_mode or random_mode or packnet_mode):
                        hist['alpha_ltp'] = alpha_ltp
                        hist['alpha_ltd'] = alpha_ltd
                    hist['hidden_size'] = hidden_size
                    hist['lr'] = lr
                    histories.append(hist)
                    if output_dir:
                        from src.utils import save_aggregated_results
                        save_aggregated_results(os.path.join(output_dir, f"{config_name}.json"), histories)
                runs_completed += 1
            current_seed += 1

    if histories:
        avg_class_il = np.mean([h['final_avg_acc'] for h in histories])
        avg_task_il = np.mean([h['final_avg_acc_task_il'] for h in histories])
        std_class_il = np.std([h['final_avg_acc'] for h in histories])
        std_task_il = np.std([h['final_avg_acc_task_il'] for h in histories])
    else:
        avg_class_il = avg_task_il = std_class_il = std_task_il = 0.0

    result = {
        "config_name": config_name,
        "percentile": percentile,
        "alpha_ltp": alpha_ltp,
        "alpha_ltd": alpha_ltd,
        "hidden_size": hidden_size,
        "epochs": epochs,
        "lr": lr,
        "avg_class_il": avg_class_il,
        "std_class_il": std_class_il,
        "avg_task_il": avg_task_il,
        "std_task_il": std_task_il,
        "histories": histories,
    }

    return result


def save_sweep_results(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    sweep_file = os.path.join(output_dir, "sweep_results.json")

    serializable = []
    for r in results:
        sr = {k: v for k, v in r.items() if k != 'histories'}
        serializable.append(sr)

    with open(sweep_file, 'w') as f:
        json.dump(serializable, f, indent=4)

    for r in results:
        config_file = os.path.join(output_dir, f"{r['config_name']}.json")
        from src.utils import save_aggregated_results
        save_aggregated_results(config_file, r['histories'])

    print(f"\n{'='*100}")
    print(f"{'CONFIG':<55} {'Class-IL':>12} {'Task-IL':>12} {'Runs':>5}")
    print(f"{'='*100}")

    sorted_results = sorted(results, key=lambda x: x['avg_task_il'], reverse=True)
    for r in sorted_results:
        name = r['config_name']
        cil = f"{r['avg_class_il']:.1f}±{r['std_class_il']:.1f}"
        til = f"{r['avg_task_il']:.1f}±{r['std_task_il']:.1f}"
        runs = len(r['histories'])
        print(f"{name:<55} {cil:>12} {til:>12} {runs:>5}")

    print(f"{'='*100}")
    print(f"\nResults saved to {output_dir}")
    return sorted_results


def get_sweep_configs(epochs):
    configs = []
    for percentile in [0.15, 0.20, 0.25]:
        for alpha_ltp in [0.01, 0.02]:
            for alpha_ltd in [0.005, 0.01]:
                configs.append({
                    'percentile': percentile,
                    'alpha_ltp': alpha_ltp,
                    'alpha_ltd': alpha_ltd,
                    'hidden_size': 1024,
                    'epochs': epochs,
                    'lr': 1e-3,
                })
    for percentile in [0.15, 0.20]:
        for alpha_ltp in [0.01, 0.02]:
            for alpha_ltd in [0.005, 0.01]:
                configs.append({
                    'percentile': percentile,
                    'alpha_ltp': alpha_ltp,
                    'alpha_ltd': alpha_ltd,
                    'hidden_size': 2048,
                    'epochs': epochs,
                    'lr': 1e-3,
                })
    return configs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fashion-MNIST Baseline \u0026 Sweep Runner")
    parser.add_argument("--mode", type=str, default="sweep", choices=["single", "sweep"])
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--seed_start", type=int, default=42)

    parser.add_argument("--percentile", type=float, default=0.15)
    parser.add_argument("--alpha_ltp", type=float, default=0.05)
    parser.add_argument("--alpha_ltd", type=float, default=0.005)
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_tasks", type=int, default=5, choices=[3,4,5])
    
    # Baseline flags
    parser.add_argument("--er_mode", action="store_true")
    parser.add_argument("--er_buffer", type=int, default=200)
    parser.add_argument("--baseline_mode", action="store_true")
    parser.add_argument("--ewc_mode", action="store_true")
    parser.add_argument("--ewc_lambda", type=float, default=1000.0)
    parser.add_argument("--si_mode", action="store_true")
    parser.add_argument("--si_c", type=float, default=1.0)
    parser.add_argument("--si_xi", type=float, default=0.1)
    parser.add_argument("--random_mode", action="store_true")
    parser.add_argument("--packnet_mode", action="store_true")

    args = parser.parse_args()

    output_dir = f"results/SNN/{args.num_tasks}-Split-FashionMNIST/sweep_epochs_{args.epochs}"
    os.makedirs(output_dir, exist_ok=True)

    if args.mode == "single":
        result = run_config(
            percentile=args.percentile,
            alpha_ltp=args.alpha_ltp,
            alpha_ltd=args.alpha_ltd,
            num_tasks=args.num_tasks,
            hidden_size=args.hidden_size,
            epochs=args.epochs,
            lr=args.lr,
            num_runs=args.runs,
            seed_start=args.seed_start,
            er_mode=args.er_mode,
            er_buffer=args.er_buffer,
            baseline_mode=args.baseline_mode,
            ewc_mode=args.ewc_mode,
            ewc_lambda=args.ewc_lambda,
            si_mode=args.si_mode,
            si_c=args.si_c,
            si_xi=args.si_xi,
            random_mode=args.random_mode,
            packnet_mode=args.packnet_mode,
            output_dir=output_dir,
        )
        save_sweep_results([result], output_dir)

    elif args.mode == "sweep":
        configs = get_sweep_configs(args.epochs)
        print(f"\nTotal configs to sweep: {len(configs)}")
        print(f"Runs per config: {args.runs}")
        print(f"Estimated total experiments: {len(configs) * args.runs}")

        all_results = []
        for i, cfg in enumerate(configs):
            print(f"\n>>> Config {i+1}/{len(configs)}")
            result = run_config(
                num_runs=args.runs,
                seed_start=args.seed_start,
                er_mode=args.er_mode,
                er_buffer=args.er_buffer,
                baseline_mode=args.baseline_mode,
                ewc_mode=args.ewc_mode,
                ewc_lambda=args.ewc_lambda,
                si_mode=args.si_mode,
                si_c=args.si_c,
                si_xi=args.si_xi,
                random_mode=args.random_mode,
                packnet_mode=args.packnet_mode,
                output_dir=output_dir,
                **cfg,
            )
            all_results.append(result)

            save_sweep_results(all_results, output_dir)

        sorted_results = save_sweep_results(all_results, output_dir)
        print(f"\n🏆 Best config: {sorted_results[0]['config_name']}")
        print(f"   Task-IL: {sorted_results[0]['avg_task_il']:.1f}±{sorted_results[0]['std_task_il']:.1f}")
        print(f"   Class-IL: {sorted_results[0]['avg_class_il']:.1f}±{sorted_results[0]['std_class_il']:.1f}")
