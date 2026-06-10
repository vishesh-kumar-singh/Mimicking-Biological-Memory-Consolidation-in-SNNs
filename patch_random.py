import sys

with open("scripts/run_random.py", "r") as f:
    content = f.read()

# Fix imports
if "NMNISTDatasetWrapper" not in content:
    content = content.replace("from src.dataset import SpikeMNISTDataset", "from src.dataset import SpikeMNISTDataset, NMNISTDatasetWrapper")

# Fix run_experiment signature
content = content.replace("def run_experiment(run_id, epochs, seed, percentile):", 'def run_experiment(run_id, epochs, seed, percentile, data_dir="spike_mnist_dataset", is_nmnist=False):')

# Fix dataset loading logic
old_data_loading = """    spike_file = os.path.join(DATA_DIR, "spike_trains_100ts.npy")
    label_file = os.path.join(DATA_DIR, "labels.npy")
    if not os.path.exists(spike_file):
        print("Data not found.")
        return None

    # Datasets for continual learning
    dataset_a = SpikeMNISTDataset(spike_file, label_file, target_digits=[0,1,2,3,4])
    dataset_b = SpikeMNISTDataset(spike_file, label_file, target_digits=[5,6,7,8,9])
    dataset_all = SpikeMNISTDataset(spike_file, label_file, target_digits=list(range(10)))"""

new_data_loading = """    if is_nmnist:
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
        input_dim = 784"""

content = content.replace(old_data_loading, new_data_loading)

# Fix model initialization
content = content.replace("model = SNNModelBaseline(hidden_size=HIDDEN_SIZE).to(DEVICE)", "model = SNNModelBaseline(input_size=input_dim, hidden_size=HIDDEN_SIZE).to(DEVICE)")

# Fix ckpt directory
content = content.replace('ckpt_dir = os.path.join("checkpoints", "MNIST")', 'dataset_name = "NMNIST" if is_nmnist else "MNIST"\n    ckpt_dir = os.path.join("checkpoints", dataset_name)')

# Fix argparse and output dir
old_argparse = """    parser.add_argument("--epochs", type=int, default=5, help="Epochs per task")
    parser.add_argument("--percentile", type=float, default=0.7, help="Freezing percentile")
    args = parser.parse_args()
    
    from src.utils import load_legacy_json, parse_results_file, save_aggregated_results
    
    # Output Directory (same as other results)
    percentile_int = int(args.percentile * 100)
    output_dir = f"results/results_epochs_{args.epochs}"
    os.makedirs(output_dir, exist_ok=True)
    results_file = f"{output_dir}/random_{percentile_int}.json\""""

new_argparse = """    parser.add_argument("--epochs", type=int, default=5, help="Epochs per task")
    parser.add_argument("--percentile", type=float, default=0.7, help="Freezing percentile")
    parser.add_argument("--data_dir", type=str, default="spike_mnist_dataset", help="Directory with spike data")
    parser.add_argument("--dataset_name", type=str, default="Split-MNIST", help="Name for results directory")
    parser.add_argument("--is_nmnist", action="store_true", help="Use NMNISTDatasetWrapper")
    args = parser.parse_args()
    
    from src.utils import load_legacy_json, parse_results_file, save_aggregated_results
    
    # Output Directory (same as other results)
    percentile_int = int(args.percentile * 100)
    output_dir = f"results/SNN/{args.dataset_name}/epochs_{args.epochs}"
    os.makedirs(output_dir, exist_ok=True)
    results_file = f"{output_dir}/random_{percentile_int}.json\""""

content = content.replace(old_argparse, new_argparse)

# Fix run_experiment call
content = content.replace("hist = run_experiment(len(histories), args.epochs, current_seed, args.percentile)", "hist = run_experiment(len(histories), args.epochs, current_seed, args.percentile, data_dir=args.data_dir, is_nmnist=args.is_nmnist)")

with open("scripts/run_random.py", "w") as f:
    f.write(content)

