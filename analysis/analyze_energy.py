"""
analyze_energy.py - Computational Cost & Energy Analysis for SNN Experiments

Computes SNN-specific energy metrics that are critical for reviewer response:
  1. Spike Count: Total spikes emitted per layer per sample (∝ energy on neuromorphic hardware)
  2. Synaptic Operations (SynOps): spikes × fan-out connections (actual compute cost)
  3. Spike Sparsity: Fraction of zero activations (higher = more efficient)
  4. Frozen Neuron Ratio: Capacity usage after consolidation
  5. Parameter Count & Memory: Model footprint

The key insight for SNNs: energy ∝ spike_count × fan_out (SynOps), NOT total parameters.
Frozen neurons that don't spike consume ZERO energy on neuromorphic hardware.

Usage:
------
    # Analyze a specific checkpoint
    python analysis/analyze_energy.py --checkpoint checkpoints/MNIST/seed_42_epochs_3_taskA.pt

    # Analyze across all checkpoints in a directory
    python analysis/analyze_energy.py --checkpoint_dir checkpoints/MNIST/

    # Run fresh inference on test data and collect spike stats  
    python analysis/analyze_energy.py --epochs 3 --percentile 0.8 --seed 42

    # Compare baseline vs freezing energy profile
    python analysis/analyze_energy.py --compare --epochs 3 --percentile 0.8
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import json
import argparse
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.dataset import SpikeMNISTDataset
from src.models import SNNModelLTP_LTD, update_p_factor_combined
from src.models.baseline import SNNModelBaseline

# =============================================================================
# Hyperparameters (must match training scripts)
# =============================================================================
BATCH_SIZE = 32
HIDDEN_SIZE = 1024
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "spike_mnist_dataset"


def count_spikes_inference(model, dataloader, device, time_steps=100):
    """
    Run inference and count spikes per layer per sample.
    
    Returns:
        dict with:
            - input_spikes_per_sample: mean input spikes per sample per timestep
            - layer1_spikes_per_sample: mean hidden layer spikes per sample per timestep
            - layer2_spikes_per_sample: mean output layer spikes per sample per timestep
            - layer1_sparsity: fraction of zero hidden activations
            - layer2_sparsity: fraction of zero output activations
            - total_samples: number of samples processed
            - layer1_spike_hist: histogram of per-neuron spike counts
    """
    model.eval()
    
    total_input_spikes = 0
    total_l1_spikes = 0
    total_l2_spikes = 0
    total_l1_possible = 0  # total possible spikes (neurons × timesteps × samples)
    total_l2_possible = 0
    total_input_possible = 0
    total_samples = 0
    
    # Per-neuron spike accumulator for histogram
    l1_size = model.layer1.linear.out_features
    l2_size = model.layer2.linear.out_features
    input_size = model.layer1.linear.in_features
    neuron_spike_counts = torch.zeros(l1_size, device=device)
    
    with torch.no_grad():
        for spikes_batch, labels in dataloader:
            spikes_batch = spikes_batch.to(device)
            batch_size = spikes_batch.size(0)
            total_samples += batch_size
            
            # Count input spikes
            total_input_spikes += spikes_batch.sum().item()
            total_input_possible += batch_size * time_steps * input_size
            
            # Manual forward pass to count intermediate spikes
            model.reset()
            
            batch_l1_spikes = 0
            batch_l2_spikes = 0
            
            for t in range(time_steps):
                input_t = spikes_batch[:, t, :]
                
                # Layer 1
                if hasattr(model.layer1, 'scale_weights') and model.layer1.scale_weights:
                    effective_weight = model.layer1.linear.weight * (1 + model.layer1.P.unsqueeze(1))
                    I1 = torch.nn.functional.linear(input_t, effective_weight)
                else:
                    I1 = model.layer1.linear(input_t)
                spikes1 = model.layer1.neuron(I1)
                
                # Layer 2  
                if hasattr(model.layer2, 'scale_weights') and model.layer2.scale_weights:
                    effective_weight2 = model.layer2.linear.weight * (1 + model.layer2.P.unsqueeze(1))
                    I2 = torch.nn.functional.linear(spikes1, effective_weight2)
                else:
                    I2 = model.layer2.linear(spikes1)
                spikes2 = model.layer2.neuron(I2)
                
                batch_l1_spikes += spikes1.sum().item()
                batch_l2_spikes += spikes2.sum().item()
                
                # Accumulate per-neuron counts
                neuron_spike_counts += spikes1.sum(dim=0)
            
            total_l1_spikes += batch_l1_spikes
            total_l2_spikes += batch_l2_spikes
            total_l1_possible += batch_size * time_steps * l1_size
            total_l2_possible += batch_size * time_steps * l2_size
    
    results = {
        "total_samples": total_samples,
        "time_steps": time_steps,
        
        # Architecture info
        "input_size": input_size,
        "hidden_size": l1_size,
        "output_size": l2_size,
        
        # Raw spike counts (total across all samples)
        "total_input_spikes": total_input_spikes,
        "total_layer1_spikes": total_l1_spikes,
        "total_layer2_spikes": total_l2_spikes,
        
        # Per-sample averages
        "input_spikes_per_sample": total_input_spikes / total_samples,
        "layer1_spikes_per_sample": total_l1_spikes / total_samples,
        "layer2_spikes_per_sample": total_l2_spikes / total_samples,
        
        # Sparsity (fraction of ZERO activations = energy savings)
        "layer1_sparsity": 1.0 - (total_l1_spikes / total_l1_possible) if total_l1_possible > 0 else 0,
        "layer2_sparsity": 1.0 - (total_l2_spikes / total_l2_possible) if total_l2_possible > 0 else 0,
        "input_sparsity": 1.0 - (total_input_spikes / total_input_possible) if total_input_possible > 0 else 0,
        
        # Synaptic Operations (SynOps) = spikes × fan-out
        # This is THE key energy metric for neuromorphic hardware
        "synops_layer1": total_l1_spikes * l2_size,  # each L1 spike triggers L2_size synapses
        "synops_layer2": total_l2_spikes * 1,         # output layer (no fan-out)
        "synops_input": total_input_spikes * l1_size,  # each input spike triggers L1_size synapses
        "synops_total_per_sample": (total_input_spikes * l1_size + total_l1_spikes * l2_size) / total_samples,
        
        # Per-neuron spike distribution (for histogram plots)
        "neuron_spike_counts_mean": (neuron_spike_counts / total_samples).cpu().tolist(),
    }
    
    return results


def analyze_frozen_capacity(model):
    """
    Analyze the frozen vs plastic neuron ratio and capacity usage.
    
    Returns:
        dict with frozen/plastic neuron counts and P-factor statistics
    """
    results = {}
    
    if hasattr(model, 'layer1') and hasattr(model.layer1, 'P'):
        p1 = model.layer1.P.detach().cpu()
        results["layer1_p_mean"] = p1.mean().item()
        results["layer1_p_std"] = p1.std().item()
        results["layer1_p_min"] = p1.min().item()
        results["layer1_p_max"] = p1.max().item()
        results["layer1_p_median"] = p1.median().item()
        
        # Count neurons at various P thresholds
        results["layer1_p_gt_0.1"] = (p1 > 0.1).sum().item()
        results["layer1_p_gt_0.3"] = (p1 > 0.3).sum().item()
        results["layer1_p_gt_0.5"] = (p1 > 0.5).sum().item()
        results["layer1_p_gt_0.7"] = (p1 > 0.7).sum().item()
        results["layer1_p_gt_0.9"] = (p1 > 0.9).sum().item()
        results["layer1_total_neurons"] = p1.size(0)
        
        # P-factor distribution (10 bins)
        hist, bin_edges = np.histogram(p1.numpy(), bins=10, range=(0, 1))
        results["layer1_p_histogram"] = hist.tolist()
        results["layer1_p_bin_edges"] = bin_edges.tolist()
    
    if hasattr(model, 'layer2') and hasattr(model.layer2, 'P'):
        p2 = model.layer2.P.detach().cpu()
        results["layer2_p_mean"] = p2.mean().item()
        results["layer2_p_std"] = p2.std().item()
    
    return results


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Memory in bytes (float32 = 4 bytes per param)
    memory_bytes = total * 4
    
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "memory_bytes": memory_bytes,
        "memory_kb": memory_bytes / 1024,
        "memory_mb": memory_bytes / (1024 * 1024),
    }


def run_energy_analysis(seed, epochs, percentile, data_dir=DATA_DIR, is_nmnist=False):
    """
    Run a full energy analysis: train Task A, consolidate, train Task B,
    then measure spike counts at each stage.
    """
    from experiments.run_freezing import run_experiment, generate_static_mask_and_reset
    
    print(f"\n{'='*60}")
    print(f"ENERGY ANALYSIS (Seed {seed}, Epochs {epochs}, τ={percentile})")
    print(f"{'='*60}")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Load data
    if is_nmnist:
        from src.dataset import NMNISTDatasetWrapper
        dataset_a = NMNISTDatasetWrapper(save_to=data_dir, train=True, target_digits=[0,1,2,3,4])
        dataset_b = NMNISTDatasetWrapper(save_to=data_dir, train=True, target_digits=[5,6,7,8,9])
        dataset_all = NMNISTDatasetWrapper(save_to=data_dir, train=True, target_digits=list(range(10)))
        input_dim = 2312
    else:
        spike_file = os.path.join(data_dir, "spike_trains_100ts.npy")
        label_file = os.path.join(data_dir, "labels.npy")
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
    
    # Try to load cached Task A checkpoint
    dataset_name = "NMNIST" if is_nmnist else "MNIST"
    ckpt_dir = os.path.join("checkpoints", dataset_name)
    ckpt_path = os.path.join(ckpt_dir, f"seed_{seed}_epochs_{epochs}_taskA.pt")
    
    model = SNNModelLTP_LTD(input_size=input_dim, hidden_size=HIDDEN_SIZE).to(DEVICE)
    
    if os.path.exists(ckpt_path):
        print(f"Loading Task A checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        print("No checkpoint found. Training Task A from scratch...")
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        from tqdm import tqdm
        for epoch in range(epochs):
            model.train()
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
                update_p_factor_combined(model, preds, l, alpha_ltp=0.01)
        # Save checkpoint
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)
    
    # ---- Stage 1: After Task A, before consolidation ----
    print("\n--- Stage 1: After Task A (pre-consolidation) ---")
    stage1_spikes = count_spikes_inference(model, test_a, DEVICE)
    stage1_capacity = analyze_frozen_capacity(model)
    stage1_params = count_parameters(model)
    
    print(f"  Input spikes/sample:  {stage1_spikes['input_spikes_per_sample']:.1f}")
    print(f"  Hidden spikes/sample: {stage1_spikes['layer1_spikes_per_sample']:.1f}")
    print(f"  Output spikes/sample: {stage1_spikes['layer2_spikes_per_sample']:.1f}")
    print(f"  Hidden sparsity:      {stage1_spikes['layer1_sparsity']*100:.1f}%")
    print(f"  SynOps/sample:        {stage1_spikes['synops_total_per_sample']:.0f}")
    print(f"  P > 0.5:              {stage1_capacity.get('layer1_p_gt_0.5', 'N/A')}/{HIDDEN_SIZE}")
    print(f"  Parameters:           {stage1_params['total_parameters']:,}")
    print(f"  Memory:               {stage1_params['memory_kb']:.1f} KB")
    
    # ---- Stage 2: After consolidation (freeze + reset) ----
    print(f"\n--- Stage 2: After consolidation (τ={percentile}) ---")
    # Deep copy model state for consolidation
    import copy
    model_consolidated = copy.deepcopy(model)
    static_masks = generate_static_mask_and_reset(model_consolidated, threshold_percentile=percentile)
    
    stage2_spikes = count_spikes_inference(model_consolidated, test_a, DEVICE)
    stage2_capacity = analyze_frozen_capacity(model_consolidated)
    
    frozen_count = int(HIDDEN_SIZE * percentile)
    plastic_count = HIDDEN_SIZE - frozen_count
    
    print(f"  Hidden spikes/sample: {stage2_spikes['layer1_spikes_per_sample']:.1f}")
    print(f"  Hidden sparsity:      {stage2_spikes['layer1_sparsity']*100:.1f}%")
    print(f"  SynOps/sample:        {stage2_spikes['synops_total_per_sample']:.0f}")
    print(f"  Frozen neurons:       {frozen_count}/{HIDDEN_SIZE}")
    print(f"  Plastic neurons:      {plastic_count}/{HIDDEN_SIZE}")
    print(f"  Capacity remaining:   {plastic_count/HIDDEN_SIZE*100:.1f}%")
    
    # ---- Stage 3: After Task B training ----
    print(f"\n--- Stage 3: Training Task B with frozen neurons ---")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_consolidated.parameters(), lr=LR)
    from tqdm import tqdm
    
    for epoch in range(epochs):
        model_consolidated.train()
        pbar = tqdm(train_b, desc=f"Task B Epoch {epoch+1}")
        for s, l in pbar:
            s, l = s.to(DEVICE), l.to(DEVICE)
            optimizer.zero_grad()
            model_consolidated.reset()
            out = model_consolidated(s)
            loss = criterion(out, l)
            loss.backward()
            # Apply gradient masks
            if static_masks:
                if model_consolidated.layer1.linear.weight.grad is not None:
                    model_consolidated.layer1.linear.weight.grad.data.mul_(static_masks['layer1'])
                if model_consolidated.layer2.linear.weight.grad is not None:
                    model_consolidated.layer2.linear.weight.grad.data.mul_(static_masks['layer2'])
            torch.nn.utils.clip_grad_norm_(model_consolidated.parameters(), max_norm=1.0)
            optimizer.step()
    
    # Measure spikes on both tasks after Task B
    stage3_spikes_a = count_spikes_inference(model_consolidated, test_a, DEVICE)
    stage3_spikes_b = count_spikes_inference(model_consolidated, test_b, DEVICE)
    stage3_spikes_all = count_spikes_inference(model_consolidated, test_all, DEVICE)
    
    print(f"  Task A spikes/sample: {stage3_spikes_a['layer1_spikes_per_sample']:.1f}")
    print(f"  Task B spikes/sample: {stage3_spikes_b['layer1_spikes_per_sample']:.1f}")
    print(f"  All spikes/sample:    {stage3_spikes_all['layer1_spikes_per_sample']:.1f}")
    print(f"  Task A SynOps/sample: {stage3_spikes_a['synops_total_per_sample']:.0f}")
    print(f"  Task B SynOps/sample: {stage3_spikes_b['synops_total_per_sample']:.0f}")
    print(f"  All SynOps/sample:    {stage3_spikes_all['synops_total_per_sample']:.0f}")
    print(f"  Task A sparsity:      {stage3_spikes_a['layer1_sparsity']*100:.1f}%")
    print(f"  Task B sparsity:      {stage3_spikes_b['layer1_sparsity']*100:.1f}%")
    
    # ---- Compile full results ----
    results = {
        "config": {
            "seed": seed,
            "epochs": epochs,
            "percentile": percentile,
            "hidden_size": HIDDEN_SIZE,
            "dataset": dataset_name,
        },
        "parameters": stage1_params,
        "stage1_after_task_a": {
            "spikes": stage1_spikes,
            "capacity": stage1_capacity,
        },
        "stage2_after_consolidation": {
            "spikes": stage2_spikes,
            "capacity": stage2_capacity,
            "frozen_neurons": frozen_count,
            "plastic_neurons": plastic_count,
        },
        "stage3_after_task_b": {
            "task_a_spikes": stage3_spikes_a,
            "task_b_spikes": stage3_spikes_b,
            "all_spikes": stage3_spikes_all,
        },
        "energy_summary": {
            "synops_per_sample_task_a": stage3_spikes_a["synops_total_per_sample"],
            "synops_per_sample_task_b": stage3_spikes_b["synops_total_per_sample"],
            "synops_per_sample_all": stage3_spikes_all["synops_total_per_sample"],
            "sparsity_task_a": stage3_spikes_a["layer1_sparsity"],
            "sparsity_task_b": stage3_spikes_b["layer1_sparsity"],
            "frozen_ratio": frozen_count / HIDDEN_SIZE,
            "capacity_remaining": plastic_count / HIDDEN_SIZE,
        }
    }
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SNN Energy & Computational Cost Analysis")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=3, help="Epochs per task")
    parser.add_argument("--percentile", type=float, default=0.8, help="Freezing percentile")
    parser.add_argument("--data_dir", type=str, default="spike_mnist_dataset", help="Data directory")
    parser.add_argument("--is_nmnist", action="store_true", help="Use N-MNIST dataset")
    parser.add_argument("--compare", action="store_true", help="Compare multiple percentiles")
    args = parser.parse_args()
    
    dataset_name = "Split-NMNIST" if args.is_nmnist else "Split-MNIST"
    output_dir = f"results/SNN/{dataset_name}/energy"
    os.makedirs(output_dir, exist_ok=True)
    
    if args.compare:
        # Compare across percentiles
        output_file = f"{output_dir}/energy_comparison_seed{args.seed}_epochs{args.epochs}.json"
        if os.path.exists(output_file):
            print(f"\nSkipping: {output_file} already exists.")
            sys.exit(0)
            
        all_results = {}
        for p in [0.4, 0.6, 0.8]:
            results = run_energy_analysis(
                args.seed, args.epochs, p,
                data_dir=args.data_dir, is_nmnist=args.is_nmnist
            )
            all_results[f"percentile_{int(p*100)}"] = results
        
        output_file = f"{output_dir}/energy_comparison_seed{args.seed}_epochs{args.epochs}.json"
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=4)
        print(f"\nComparison saved to {output_file}")
        
        # Print summary table
        print(f"\n{'='*70}")
        print(f"ENERGY COMPARISON SUMMARY (Seed {args.seed}, Epochs {args.epochs})")
        print(f"{'='*70}")
        print(f"{'Percentile':>12} {'SynOps/sample':>15} {'Sparsity':>10} {'Frozen':>8} {'Capacity':>10}")
        print(f"{'-'*70}")
        for key, res in all_results.items():
            es = res["energy_summary"]
            print(f"{key:>12} {es['synops_per_sample_all']:>15,.0f} "
                  f"{es['sparsity_task_a']*100:>9.1f}% "
                  f"{es['frozen_ratio']*100:>7.0f}% "
                  f"{es['capacity_remaining']*100:>9.0f}%")
    else:
        # Single analysis
        p_int = int(args.percentile * 100)
        output_file = f"{output_dir}/energy_seed{args.seed}_epochs{args.epochs}_p{p_int}.json"
        if os.path.exists(output_file):
            print(f"\nSkipping: {output_file} already exists.")
            sys.exit(0)
            
        results = run_energy_analysis(
            args.seed, args.epochs, args.percentile,
            data_dir=args.data_dir, is_nmnist=args.is_nmnist
        )
        
        p_int = int(args.percentile * 100)
        output_file = f"{output_dir}/energy_seed{args.seed}_epochs{args.epochs}_p{p_int}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\nResults saved to {output_file}")
