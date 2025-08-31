import subprocess
import sys
import numpy as np
import pandas as pd
from skopt import gp_minimize
from skopt.space import Real
import os

WORKDIR = r"E:\PC\ABAQUS_SIM\R_D\COMTES_AM\Simulation\Bayesian"
WORKER_SCRIPT = r"E:\PC\ABAQUS_SIM\R_D\COMTES_AM\Simulation\Bayesian\worker.py"
ABAQUS_CMD = r"C:\SIMULIA\Commands\abq2024.bat"

def run_worker(n_val):
    # Call Abaqus with worker.py and n_val as argument
    cmd = [ABAQUS_CMD, 'cae', 'noGUI={}'.format(WORKER_SCRIPT), '--', str(n_val)]
    print(">> Launching:", ' '.join(cmd))
    proc = subprocess.run(cmd, cwd=WORKDIR)
    # Read result from file
    objfile = os.path.join(WORKDIR, "objective.txt")
    if not os.path.exists(objfile):
        print("### objective.txt not found, assigning inf")
        return 1e6
    try:
        with open(objfile, "r") as f:
            val = float(f.read().strip())
    except Exception:
        val = 1e6
    return val

results = []
def objective(n):
    n_val = n[0]
    cost = run_worker(n_val)
    results.append({'n': n_val, 'Cost Function': cost})
    return cost

space = [Real(-0.5, 1.5, name='n')]
opt_result = gp_minimize(objective, space, n_calls=15, n_initial_points=3, random_state=42)

# Save results
df = pd.DataFrame(results)
df.insert(0, "Iteration", range(1, 1 + len(df)))
df.to_excel(os.path.join(WORKDIR, "Optimization_Results_Bayesian.xlsx"), index=False)
print("Best n:", opt_result.x[0], "Cost Function:", opt_result.fun)
print("Results saved.")
