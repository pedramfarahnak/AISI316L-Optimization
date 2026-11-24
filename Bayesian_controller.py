"""Bayesian BO controller for Abaqus 2024; run using ordinary Python from CMD.

Requires: python -m pip install numpy pandas scipy scikit-optimize openpyxl
Worker runs under Abaqus CAE's Python and uses the V7/V8 validated 40-point
hardening formulation (only execution and bookkeeping changed).
"""
from pathlib import Path
from datetime import datetime
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from skopt import gp_minimize
from skopt.space import Real

WORKDIR = Path(r"E:\PC\ABAQUS_SIM\R_D\COMTES_AM\Simulation\Paper\Drawbacks")
ABAQUS_CMD = r"C:\SIMULIA\Commands\abq2024.bat"
WORKER_SCRIPT = WORKDIR / "Bayesian_worker.py"

# ============ USER SETTINGS ============
SINGLE_TEST = False       # True: one FE job with TEST_N; False: Bayesian optimization
TEST_N = 1.0
N_CALLS = 15
N_INITIAL_POINTS = 3
RANDOM_SEED = 42
N_BOUNDS = (0.60, 1.40)  # Deliberate range: same for all optimizers in paper!
# To search the former full domain instead, set (0.05, 1.50) here;
# do not make superiority claims based on a narrowed range without disclosure.
# =======================================


def new_run_directory():
    root = WORKDIR / "BO_runs"
    root.mkdir(exist_ok=True)
    base = datetime.now().strftime("BO_%Y%m%d_%H%M%S")
    for i in range(1000):
        name = base if i == 0 else f"{base}_{i:03d}"
        path = root / name
        try:
            path.mkdir()
            return name, path
        except FileExistsError:
            continue
    raise RuntimeError("Unable to allocate a unique run directory")


def call_worker(n, index, run_id, run_dir):
    env = os.environ.copy()
    env["BAYESIAN_N"] = repr(float(n))
    env["BAYESIAN_EVAL_ID"] = str(index)
    env["BAYESIAN_RUN_ID"] = run_id
    output_path = run_dir / f"objective_{index:04d}.json"
    log_path = run_dir / f"abaqus_launch_{index:04d}.log"
    cmd = [ABAQUS_CMD, "cae", f"noGUI={WORKER_SCRIPT}"]
    print(f"Evaluation {index}: n={n:.8f} | Starting Abaqus...", flush=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        result = subprocess.run(cmd, cwd=str(run_dir), env=env,
                                stdout=log, stderr=subprocess.STDOUT, check=False)
    if not output_path.is_file():
        raise RuntimeError(f"No objective JSON from worker. See {log_path}")
    with output_path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    if result.returncode != 0 or data.get("status") != "ok":
        raise RuntimeError(
            f"FE evaluation {index} FAILED: {data.get('error', 'see log')} "
            f"(exit={result.returncode}). See {log_path} and worker_error_{index:04d}.log"
        )
    value = float(data["nrmse_percent"])
    if not np.isfinite(value) or value < 0:
        raise RuntimeError(f"Invalid NRMSE from worker: {value}")
    print(f"  -> Post-neck NRMSE: {value:.6f}%", flush=True)
    return value


def main():
    for required in (WORKER_SCRIPT, WORKDIR / "UT_YXZ_M.inp",
                     WORKDIR / "Tensile_test.xls", WORKDIR / "test.xlsx"):
        if not required.is_file():
            raise FileNotFoundError(required)
    if not (0 < N_BOUNDS[0] < N_BOUNDS[1]):
        raise ValueError("Invalid n bounds")
    if N_CALLS <= N_INITIAL_POINTS:
        raise ValueError("N_CALLS must exceed N_INITIAL_POINTS")
    run_id, run_dir = new_run_directory()
    print(f"Run directory: {run_dir}")
    config = {"single_test": SINGLE_TEST, "test_n": TEST_N,
              "n_bounds": N_BOUNDS, "n_calls": N_CALLS,
              "initial_points": N_INITIAL_POINTS, "seed": RANDOM_SEED,
              "objective": "fixed-range post-neck NRMSE (%), 125 uniform u points",
              "experimental_sheet": "Results",
              "hardening_points": 40,
              "reference_worker": "V7 hardening equations"}
    (run_dir / "configuration.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    history = []

    def evaluate(x):
        n = float(x[0])
        i = len(history) + 1
        try:
            value = call_worker(n, i, run_id, run_dir)
        except Exception as exc:
            history.append({"evaluation": i, "n": n,
                            "PostNeck_NRMSE_percent": np.nan,
                            "status": "failed", "error": str(exc)})
            pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
            raise  # Stop BO rather than falsely awarding an invalid FE penalty
        history.append({"evaluation": i, "n": n,
                        "PostNeck_NRMSE_percent": value, "status": "ok"})
        df = pd.DataFrame(history)
        df.to_csv(run_dir / "history.csv", index=False)
        df.to_excel(run_dir / "Optimization_Results_Bayesian.xlsx", index=False)
        return value

    if SINGLE_TEST:
        result = evaluate([TEST_N])
        print(f"Single FE test completed. NRMSE={result:.6f}%")
        return
    result = gp_minimize(evaluate, [Real(*N_BOUNDS, name="n")],
                         n_calls=N_CALLS, n_initial_points=N_INITIAL_POINTS,
                         random_state=RANDOM_SEED)
    best = {"best_n": float(result.x[0]), "best_nrmse_percent": float(result.fun),
            "successful_evaluations": len(history), "run_id": run_id}
    (run_dir / "best_result.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    print(f"Best n={result.x[0]:.8f}; Post-neck NRMSE={result.fun:.6f}%")
    print(f"Results: {run_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"STOPPED: {exc}", file=sys.stderr)
        sys.exit(1)
