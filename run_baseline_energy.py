import sys
import os
import torch
import json
from analysis.analyze_energy import count_spikes_inference
from src.dataset import SpikeMNISTDataset
from torch.utils.data import DataLoader

from experiments.run_baseline import run_experiment as run_ft
from experiments.run_ewc import run_experiment as run_ewc
from experiments.run_si import run_experiment as run_si
from experiments.run_er import run_experiment as run_er

seed = 42
epochs = 3
device = "cuda" if torch.cuda.is_available() else "cpu"
data_dir = "spike_mnist_dataset"
dataset_name = "Split-MNIST"

spike_file = os.path.join(data_dir, "spike_trains_100ts.npy")
label_file = os.path.join(data_dir, "labels.npy")

dataset_a = SpikeMNISTDataset(spike_file, label_file, target_digits=[0,1,2,3,4])
dataset_b = SpikeMNISTDataset(spike_file, label_file, target_digits=[5,6,7,8,9])
dataset_all = SpikeMNISTDataset(spike_file, label_file, target_digits=list(range(10)))

def get_loader(ds):
    train_size = int(0.8 * len(ds))
    train_ds, test_ds = torch.utils.data.random_split(ds, [train_size, len(ds)-train_size], generator=torch.Generator().manual_seed(seed))
    return DataLoader(test_ds, batch_size=32)

test_a = get_loader(dataset_a)
test_b = get_loader(dataset_b)
test_all = get_loader(dataset_all)

results = {}

print("Running Fine-Tuning...")
hist, model_ft = run_ft(0, epochs, seed, data_dir=data_dir)
res_ft = count_spikes_inference(model_ft, test_all, device)
print("FT SynOps:", res_ft["synops_total_per_sample"])
results["Fine-Tuning"] = res_ft

print("Running ER...")
hist, model_er = run_er(0, epochs, seed, 200, data_dir=data_dir)
res_er = count_spikes_inference(model_er, test_all, device)
print("ER SynOps:", res_er["synops_total_per_sample"])
results["ER"] = res_er

print("Running EWC...")
hist, model_ewc = run_ewc(0, epochs, seed, 100000.0, data_dir=data_dir)
res_ewc = count_spikes_inference(model_ewc, test_all, device)
print("EWC SynOps:", res_ewc["synops_total_per_sample"])
results["EWC"] = res_ewc

print("Running SI...")
hist, model_si = run_si(0, epochs, seed, 1000000.0, data_dir=data_dir)
res_si = count_spikes_inference(model_si, test_all, device)
print("SI SynOps:", res_si["synops_total_per_sample"])
results["SI"] = res_si

output_dir = f"results/SNN/{dataset_name}/energy"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, f"baseline_energy_comparison_seed{seed}_epochs{epochs}.json")

with open(output_file, 'w') as f:
    json.dump(results, f, indent=4)
    
print(f"Saved results to {output_file}")
