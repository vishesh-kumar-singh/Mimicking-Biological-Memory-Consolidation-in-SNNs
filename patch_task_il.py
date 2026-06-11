import os
import glob
import re

scripts = glob.glob("scripts/run_*.py")

def patch_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Skip if already patched
    if '"full_curve_task_il"' in content:
        return False

    # 1. Update history dict
    content = re.sub(
        r'("full_curve": \[\],\n\s*)("task_b": \[\])',
        r'\1"full_curve_task_il": [],\n        \2,\n        "task_b_task_il": []',
        content
    )

    # 2. Patch loaded checkpoint print
    content = re.sub(
        r'(if start_epoch > 0:\s*\n\s*acc = evaluate\(model, test_a, DEVICE\)\s*\n\s*)for _ in range\(start_epoch\):\s*\n\s*history\["full_curve"\]\.append\(acc\)\s*\n\s*print\(f"Loaded Checkpoint Test Acc: \{acc:.2f\}%"\)',
        r'\1acc_task_il = evaluate(model, test_a, DEVICE, task_classes=[0,1,2,3,4])\n        for _ in range(start_epoch):\n            history["full_curve"].append(acc)\n            history["full_curve_task_il"].append(acc_task_il)\n            \n        print(f"Loaded Checkpoint Test Acc (Class-IL): {acc:.2f}% | (Task-IL): {acc_task_il:.2f}%")',
        content
    )

    # 3. Patch Phase 1 epoch eval
    content = re.sub(
        r'acc = evaluate\(model, test_a, DEVICE\)\n(\s*)history\["full_curve"\]\.append\(acc\)\n(\s*)print\(f"Epoch \{epoch\+1\} Test Acc: \{acc:\.2f\}%"\)',
        r'acc = evaluate(model, test_a, DEVICE)\n\1acc_task_il = evaluate(model, test_a, DEVICE, task_classes=[0,1,2,3,4])\n\1history["full_curve"].append(acc)\n\1history["full_curve_task_il"].append(acc_task_il)\n\2print(f"Epoch {epoch+1} Test Acc (Class-IL): {acc:.2f}% | (Task-IL): {acc_task_il:.2f}%")',
        content
    )

    # 4. Patch Phase 2 Task A retention
    content = re.sub(
        r'acc_retention = evaluate\(model, test_a, DEVICE\)\n(\s*)history\["full_curve"\]\.append\(acc_retention\)\n(\s*)print\(f"Epoch \{epoch\+1\} Task A Retention: \{acc_retention:\.2f\}%"\)',
        r'acc_retention = evaluate(model, test_a, DEVICE)\n\1acc_retention_task_il = evaluate(model, test_a, DEVICE, task_classes=[0,1,2,3,4])\n\1history["full_curve"].append(acc_retention)\n\1history["full_curve_task_il"].append(acc_retention_task_il)\n\2print(f"Epoch {epoch+1} Task A Retention (Class-IL): {acc_retention:.2f}% | (Task-IL): {acc_retention_task_il:.2f}%")',
        content
    )

    # 5. Patch Phase 2 Task B acc
    content = re.sub(
        r'acc_b = evaluate\(model, test_b, DEVICE\)\n(\s*)history\["task_b"\]\.append\(acc_b\)\n(\s*)print\(f"Epoch \{epoch\+1\} Task B Accuracy: \{acc_b:\.2f\}%"\)',
        r'acc_b = evaluate(model, test_b, DEVICE)\n\1acc_b_task_il = evaluate(model, test_b, DEVICE, task_classes=[5,6,7,8,9])\n\1history["task_b"].append(acc_b)\n\1history["task_b_task_il"].append(acc_b_task_il)\n\2print(f"Epoch {epoch+1} Task B Accuracy (Class-IL): {acc_b:.2f}% | (Task-IL): {acc_b_task_il:.2f}%")',
        content
    )

    with open(filepath, 'w') as f:
        f.write(content)
        
    return True

for script in scripts:
    if patch_file(script):
        print(f"Patched {script}")
    else:
        print(f"Skipped {script}")
