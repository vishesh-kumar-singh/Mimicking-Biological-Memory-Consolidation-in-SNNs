import os
import glob
import re

scripts = [
    "run_freezing.py",
    "run_noreset.py",
    "run_noscale.py",
    "run_reset_variants.py",
    "run_phase_b_p_factor.py",
]

for script in scripts:
    path = os.path.join("scripts", script)
    if not os.path.exists(path): continue
    with open(path, "r") as f:
        content = f.read()

    # Change checkpoint name
    content = content.replace('f"seed_{seed}_epochs_{epochs}_taskA.pt"', 'f"seed_{seed}_epochs_{epochs}_ltp{alpha_ltp}_ltd{alpha_ltd}_taskA.pt"')
    content = content.replace('f"seed_{seed}_epochs_{e}_taskA.pt"', 'f"seed_{seed}_epochs_{e}_ltp{alpha_ltp}_ltd{alpha_ltd}_taskA.pt"')
    content = content.replace('f"seed_{seed}_epochs_{epoch+1}_taskA.pt"', 'f"seed_{seed}_epochs_{epoch+1}_ltp{alpha_ltp}_ltd{alpha_ltd}_taskA.pt"')
    
    with open(path, "w") as f:
        f.write(content)
    print(f"Updated {script}")

