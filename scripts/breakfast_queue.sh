#!/bin/bash
echo "Waiting for run_noreset.py to finish..."
while pgrep -f run_noreset.py > /dev/null; do
    sleep 30
done

echo "run_noreset.py finished! Starting Zero variant..."
python scripts/run_reset_variants.py --epochs 3 --percentile 0.8 --runs 5 --reset_type zero

echo "Zero variant finished! Starting Scale variant..."
python scripts/run_reset_variants.py --epochs 3 --percentile 0.8 --runs 5 --reset_type scale

echo "All Task 3 experiments completed!"
