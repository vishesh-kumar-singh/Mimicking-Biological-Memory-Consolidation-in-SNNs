import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.dataset import NMNISTDatasetWrapper
from src.models import SNNModelLTP_LTD, update_p_factor_combined
from scripts.run_freezing import generate_static_mask_and_reset, evaluate

BATCH_SIZE = 32
INPUT_SIZE = 2312  # N-MNIST dimension: 34*34*2
HIDDEN_SIZE = 1024
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def run_nmnist_experiment(run_id, epochs, seed, percentile):
    print(f"\n{'='*40}")
    print(f"N-MNIST RUN {run_id+1} (Seed {seed}, Percentile {percentile})")
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
    
    # Initialize datasets using tonic wrapper
    print("Loading N-MNIST Dataset...")
    dataset_a = NMNISTDatasetWrapper(save_to='./data', train=True, target_digits=[0,1,2,3,4])
    dataset_b = NMNISTDatasetWrapper(save_to='./data', train=True, target_digits=[5,6,7,8,9])
    dataset_all = NMNISTDatasetWrapper(save_to='./data', train=False)
    
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
    
    # We use the full test set of NMNIST for final evaluation
    test_all = DataLoader(dataset_all, batch_size=BATCH_SIZE)
    
    model = SNNModelLTP_LTD(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    history = {
        "full_curve": [],
        "task_b": [],
        "eval_all": 0.0,
        "final_task_a": 0.0
    }

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
            
            update_p_factor_combined(model, preds, l, alpha_ltp=0.01)
            pbar.set_postfix({"Loss": loss.item(), "Acc": total_correct/total_samples})
                
        acc = evaluate(model, test_a, DEVICE)
        history["full_curve"].append(acc)
        print(f"Epoch {epoch+1} Test Acc: {acc:.2f}%")

    print(f"\n[System] Consolidating Memory (Percentile {percentile})...")
    static_masks = generate_static_mask_and_reset(model, threshold_percentile=percentile)

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
        
    acc_all = evaluate(model, test_all, DEVICE)
    print(f"Combined Test Accuracy: {acc_all:.2f}%")
    
    history["eval_all"] = acc_all
    history["final_task_a"] = history["full_curve"][-1]
    
    return history

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--percentile", type=float, default=0.7)
    args = parser.parse_args()
    
    from src.utils import parse_results_file, save_aggregated_results
    
    output_dir = f"results/SNN/Split-NMNIST/epochs_{args.epochs}"
    os.makedirs(output_dir, exist_ok=True)
    results_file = f"{output_dir}/freezing_{int(args.percentile*100)}.json"
    
    histories = parse_results_file(results_file)
    
    existing_seeds = {h['seed'] for h in histories if 'seed' in h}
    runs_needed = args.runs - len(histories)
    
    if runs_needed <= 0:
        print("Runs already completed. Exiting.")
        sys.exit(0)
        
    current_seed = 42
    runs_completed = 0
    
    while runs_completed < runs_needed:
        if current_seed not in existing_seeds:
            hist = run_nmnist_experiment(len(histories), args.epochs, current_seed, args.percentile)
            if hist:
                hist['seed'] = current_seed
                histories.append(hist)
                save_aggregated_results(results_file, histories)
            runs_completed += 1
        current_seed += 1
