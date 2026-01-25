import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np
import sys
import json
import argparse

                                  
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.dataset import SpikeMNISTDataset
from src.models import SNNModelBaseline

                       
BATCH_SIZE = 32
HIDDEN_SIZE = 1024
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "spike_mnist_dataset"

def evaluate(model, dataloader, device):
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

def run_experiment(run_id, epochs, seed):
    print(f"\n{'='*40}")
    print(f"RUN {run_id+1} (Seed {seed})")
    print(f"{'='*40}")
    
              
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
                
    spike_file = os.path.join(DATA_DIR, "spike_trains_100ts.npy")
    label_file = os.path.join(DATA_DIR, "labels.npy")
    if not os.path.exists(spike_file):
        print("Data not found.")
        return None

              
    dataset_a = SpikeMNISTDataset(spike_file, label_file, target_digits=[0,1,2,3,4])
    dataset_b = SpikeMNISTDataset(spike_file, label_file, target_digits=[5,6,7,8,9])
    dataset_all = SpikeMNISTDataset(spike_file, label_file, target_digits=list(range(10)))
    
    def get_loader(ds):
        train_size = int(0.8 * len(ds))
        train_ds, test_ds = torch.utils.data.random_split(ds, [train_size, len(ds)-train_size], generator=torch.Generator().manual_seed(seed))
        return DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True), DataLoader(test_ds, batch_size=BATCH_SIZE)

    train_a, test_a = get_loader(dataset_a)
    train_b, test_b = get_loader(dataset_b)
    _, test_all = get_loader(dataset_all)
    
                
    model = SNNModelBaseline(hidden_size=HIDDEN_SIZE).to(DEVICE)
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
        history["full_curve"].append(acc)
        print(f"Epoch {epoch+1} Test Acc: {acc:.2f}%")

    print("\n--- Phase 2: Training Task B ---")
                                
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
            loss = criterion(out, l)
            loss.backward()
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
    parser.add_argument("--runs", type=int, default=1, help="Number of runs with different seeds")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs per task")
    args = parser.parse_args()
    
    from src.utils import load_legacy_json, parse_results_file, save_aggregated_results
    
    results_file = f"results/results_epochs_{args.epochs}/cl_baseline.json"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
                              
    histories = parse_results_file(results_file)
    
                                                         
    if not histories:
        print("No aggregated results found. Checking for legacy JSON files...")
        legacy_histories = load_legacy_json("results", "baseline_run_*.json")
        if legacy_histories:
            print(f"Found {len(legacy_histories)} legacy runs. Merging...")
            histories.extend(legacy_histories)
    
                            
    start_seed_offset = len(histories)                                                
                                                  
                                                       
                                                                             
    
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
            hist = run_experiment(len(histories), args.epochs, current_seed)
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
