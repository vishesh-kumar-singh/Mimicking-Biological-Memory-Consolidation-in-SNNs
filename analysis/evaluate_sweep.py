import glob
import json
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1, help="Epochs to evaluate")
    args = parser.parse_args()

    results_dir = f"results/SNN/Split-MNIST/epochs_{args.epochs}"
    pattern = os.path.join(results_dir, "freezing_20_ltp*_ltd*.json")
    files = glob.glob(pattern)
    
    if not files:
        print(f"No results found matching {pattern}")
        return
        
    results = []
    for file in files:
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                
                # Extract ltp and ltd from filename
                filename = os.path.basename(file)
                parts = filename.replace('.json', '').split('_')
                ltp_str = parts[-2].replace('ltp', '')
                ltd_str = parts[-1].replace('ltd', '')
                
                ltp = float(ltp_str)
                ltd = float(ltd_str)
                
                if "average" in data:
                    avg_data = data["average"]
                    task_a_retention = avg_data.get("final_task_a_mean", 0)
                    eval_all = avg_data.get("eval_all_mean", 0)
                    results.append((ltp, ltd, task_a_retention, eval_all))
        except Exception as e:
            print(f"Error reading {file}: {e}")
            
    # Sort by Task A retention primarily, then Combined Accuracy
    results.sort(key=lambda x: (x[2], x[3]), reverse=True)
    
    print(f"{'LTP':<10} | {'LTD':<10} | {'Task A Retention':<18} | {'Combined Accuracy':<18}")
    print("-" * 65)
    for res in results:
        print(f"{res[0]:<10} | {res[1]:<10} | {res[2]:<16.2f} % | {res[3]:<16.2f} %")
        
    if results:
        best = results[0]
        print("\n" + "="*50)
        print(f"BEST CONFIGURATION:")
        print(f"alpha_ltp = {best[0]}")
        print(f"alpha_ltd = {best[1]}")
        print(f"Task A Retention = {best[2]:.2f}%")
        print(f"Combined Accuracy = {best[3]:.2f}%")
        print("="*50)

if __name__ == "__main__":
    main()
