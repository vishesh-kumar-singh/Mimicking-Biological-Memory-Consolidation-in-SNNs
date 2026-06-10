import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import json
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.dataset import SpikeMNISTDataset
from src.models import SNNModelLTP_LTD, update_p_factor_combined

BATCH_SIZE = 32
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "spike_fashion_mnist_dataset"
NUM_TASKS = 5

def evaluate_tasks(model, dataloaders, device, max_task_idx):
    """Evaluate model on all tasks up to max_task_idx."""
    model.eval()
    accuracies = []
    
    with torch.no_grad():
        for task_idx in range(max_task_idx + 1):
            total_correct = 0
            total_samples = 0
            loader = dataloaders[task_idx]
            
            for spikes, labels in loader:
                spikes, labels = spikes.to(device), labels.to(device)
                model.reset()
                out = model(spikes)
                
                # Restrict predictions to the classes seen so far or just current task?
                # CIL typically allows prediction over ALL seen classes
                # Valid classes so far: 0 to (max_task_idx * 2) + 1
                valid_classes = (max_task_idx + 1) * 2
                
                # Zero out probabilities for unseen classes to be fair, or let it guess among all 10?
                # Following standard single-head CIL, we usually evaluate over all 10 or all seen.
                # Let's let it predict over all 10 (hardest setting).
                preds = out.argmax(dim=1)
                
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)
                
            acc = total_correct / total_samples * 100 if total_samples > 0 else 0
            accuracies.append(acc)
            
    return accuracies

def generate_static_mask_and_reset_multitask(model, frozen_mask, task_idx, threshold_percentile=0.7):
    """
    Freeze top P-factor neurons from the remaining plastic neurons.
    task_idx: index of the task just completed (0 to 4).
    frozen_mask: boolean tensor [HIDDEN_SIZE] indicating which neurons are ALREADY frozen.
    """
    masks = {}
    
    if hasattr(model, 'layer1') and hasattr(model.layer1, 'P'):
        p1 = model.layer1.P.detach()
        plastic_indices = torch.where(~frozen_mask)[0]
        
        if len(plastic_indices) > 0:
            p1_plastic = p1[plastic_indices]
            k = int(len(plastic_indices) * threshold_percentile)
            
            if k > 0:
                threshold1 = torch.kthvalue(p1_plastic, len(plastic_indices) - k + 1).values
                # Find which plastic neurons exceed threshold
                newly_frozen = (p1 >= threshold1) & (~frozen_mask)
                frozen_mask = frozen_mask | newly_frozen
            
            
            # NOTE: We do NOT reset plastic neurons in multi-task.
            # Unlike 2-task, resetting 90% of hidden neurons would destroy the
            # internal representations that frozen output heads depend on,
            # causing all previous task accuracies to collapse.
                         
        # Return the gradient mask for layer 1 (0 for frozen, 1 for plastic)
        masks['layer1'] = (~frozen_mask).float().unsqueeze(1).to(p1.device)

    if hasattr(model, 'layer2'):
        # Freeze outputs of all tasks seen so far
        # Classes seen: 0 to (task_idx * 2) + 1
        num_seen_classes = (task_idx + 1) * 2
        
        mask2 = torch.ones(model.layer2.linear.weight.shape[0], 1).to(DEVICE)
        mask2[0:num_seen_classes] = 0.0
        masks['layer2'] = mask2
        
        # Reset P-factors for unseen classes so they start fresh
        if hasattr(model.layer2, 'P') and num_seen_classes < 10:
            with torch.no_grad():
                model.layer2.P[num_seen_classes:] = 0.0
                    
    return masks, frozen_mask

def run_experiment(run_id, epochs, seed, percentile, hidden_size=1024, checkpoint_dir="checkpoints"):
    print(f"\n{'='*40}")
    print(f"MULTI-TASK RUN {run_id+1} (Seed {seed}, Percentile {percentile}, Hidden {hidden_size})")
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
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
        
    os.makedirs(checkpoint_dir, exist_ok=True)
    task1_checkpoint_path = os.path.join(checkpoint_dir, f"seed_{seed}_epochs_{epochs}_task1.pt")
    
    spike_file = os.path.join(DATA_DIR, "spike_trains_100ts.npy")
    label_file = os.path.join(DATA_DIR, "labels.npy")
    if not os.path.exists(spike_file):
        print(f"Data not found in {DATA_DIR}.")
        return None
        
    train_loaders = []
    test_loaders = []
    
    for t in range(NUM_TASKS):
        target_digits = [t*2, t*2+1]
        ds = SpikeMNISTDataset(spike_file, label_file, target_digits=target_digits)
        train_size = int(0.8 * len(ds))
        train_ds, test_ds = torch.utils.data.random_split(
            ds, [train_size, len(ds)-train_size], 
            generator=torch.Generator().manual_seed(seed)
        )
        train_loaders.append(DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True))
        test_loaders.append(DataLoader(test_ds, batch_size=BATCH_SIZE))
        
    model = SNNModelLTP_LTD(hidden_size=hidden_size).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    
    # Store accuracy matrix: [task_trained][task_evaluated]
    acc_matrix = []
    
    frozen_mask = torch.zeros(hidden_size, dtype=torch.bool).to(DEVICE)
    static_masks = None
    
    for task_idx in range(NUM_TASKS):
        print(f"\n--- Training Task {task_idx+1}/5 (Classes {task_idx*2}-{task_idx*2+1}) ---")
        
        train_loader = train_loaders[task_idx]
        
        # Check if we can load Task 1 from checkpoint
        if task_idx == 0 and os.path.exists(task1_checkpoint_path):
            print(f"Loading Task 1 state from checkpoint: {task1_checkpoint_path}")
            model.load_state_dict(torch.load(task1_checkpoint_path, map_location=DEVICE))
        else:
            # Reset optimizer for new task
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            
            for epoch in range(epochs):
                model.train()
                total_correct = 0
                total_samples = 0
                
                pbar = tqdm(train_loader, desc=f"T{task_idx+1} Epoch {epoch+1}")
                for s, l in pbar:
                    s, l = s.to(DEVICE), l.to(DEVICE)
                    
                    optimizer.zero_grad()
                    model.reset()
                    out = model(s)
                    loss = criterion(out, l)
                    loss.backward()
                    
                    # Apply freezing masks
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
                    
                    update_p_factor_combined(model, preds, l, alpha_ltp=0.01)
                    pbar.set_postfix({"Acc": total_correct/total_samples})
            
            # Save Task 1 checkpoint for future sweeps
            if task_idx == 0:
                torch.save(model.state_dict(), task1_checkpoint_path)
                print(f"Saved Task 1 state to checkpoint: {task1_checkpoint_path}")
                
        # Evaluate after finishing task
        accuracies = evaluate_tasks(model, test_loaders, DEVICE, task_idx)
        acc_matrix.append(accuracies)
        
        print(f"Task {task_idx+1} completed. Accuracies on seen tasks:")
        for i, acc in enumerate(accuracies):
            print(f"  -> Task {i+1}: {acc:.2f}%")
            
        # Consolidate if not the last task
        if task_idx < NUM_TASKS - 1:
            print(f"[System] Consolidating Memory & Resetting Novices...")
            static_masks, frozen_mask = generate_static_mask_and_reset_multitask(
                model, frozen_mask, task_idx, threshold_percentile=percentile
            )
            print(f"Total neurons frozen so far: {frozen_mask.sum().item()} / {hidden_size}")

    final_avg = sum(acc_matrix[-1]) / len(acc_matrix[-1]) if acc_matrix else 0
    history = {
        "acc_matrix": acc_matrix,
        "final_avg_acc": final_avg,
        "eval_all": final_avg,
        "final_task_a": acc_matrix[-1][0] if acc_matrix else 0,
        "full_curve": [],
        "task_b": []
    }
    
    return history

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--percentiles", type=str, default="0.1,0.15,0.2")
    parser.add_argument("--seed_start", type=int, default=42)
    parser.add_argument("--hidden_size", type=int, default=1024)
    args = parser.parse_args()
    
    from src.utils import parse_results_file, save_aggregated_results
    
    percentiles = [float(p) for p in args.percentiles.split(",")]
    
    output_dir = f"results/SNN/Split-FashionMNIST/epochs_{args.epochs}"
    os.makedirs(output_dir, exist_ok=True)
    
    for p in percentiles:
        p_int = int(p * 100)
        results_file = f"{output_dir}/multitask_freezing_{p_int}.json"
        histories = parse_results_file(results_file)
        
        existing_seeds = {h['seed'] for h in histories if 'seed' in h}
        runs_needed = args.runs - len(histories)
        
        if runs_needed <= 0:
            print(f"Percentile {p}: Already have {len(histories)} runs. Skipping.")
            continue
            
        print(f"\nStarting evaluation for Percentile {p}. Need {runs_needed} more runs.")
        current_seed = args.seed_start
        runs_completed = 0
        
        while runs_completed < runs_needed:
            if current_seed not in existing_seeds:
                hist = run_experiment(len(histories), args.epochs, current_seed, p, hidden_size=args.hidden_size)
                if hist:
                    hist['seed'] = current_seed
                    hist['percentile'] = p
                    histories.append(hist)
                    save_aggregated_results(results_file, histories)
                runs_completed += 1
            current_seed += 1
