"""15-point uniform Grid Search for the AISI 316L Abaqus inverse-calibration study.

USES THE SAME WORKER PHYSICS / OBJECTIVE AS Bayesian V9:
  - 40-point plastic flow curve, 20 pre-neck + 20 post-neck points
  - fixed 125-point post-neck PCHIP NRMSE, experimental sheet 'Results'
  - same geometry, force/displacement scaling and load history

Run with system Python 3: python Grid_Search.py
Required on system Python: numpy, pandas, openpyxl
Abaqus/CAE must have the Python packages needed by GridSearch_worker.py.
"""
from datetime import datetime
from pathlib import Path
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

# ---------- USER SETTINGS ----------
WORKDIR = Path(r"E:\PC\ABAQUS_SIM\R_D\COMTES_AM\Simulation\Paper\Drawbacks")
ABAQUS_CMD = r"C:\SIMULIA\Commands\abq2024.bat"
WORKER_SCRIPT = WORKDIR / "GridSearch_worker.py"

SINGLE_TEST = False   # First confirm compatibility at n=1.0, then set False.
TEST_N = 1.0
N_MIN = 0.60
N_MAX = 1.40
N_EVALUATIONS = 15
# -----------------------------------


def new_run_dir():
    base = WORKDIR / "GridSearch_runs"
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("GRID_%Y%m%d_%H%M%S")
    for suffix in range(1000):
        path = base / (stamp if suffix == 0 else f"{stamp}_{suffix:03d}")
        try:
            path.mkdir()
            return path
        except FileExistsError:
            continue
    raise RuntimeError("Unable to create a unique output directory")


def run_fe(n, eval_id, run_dir):
    result_path = run_dir / f"objective_{eval_id:04d}.json"
    log_path = run_dir / f"abaqus_launch_{eval_id:04d}.log"
    if result_path.exists():
        raise RuntimeError(f"Unexpected pre-existing result: {result_path}")
    env = os.environ.copy()
    env.update({
        "GRID_N": repr(float(n)),
        "GRID_EVAL_ID": str(eval_id),
        "GRID_RUN_ID": run_dir.name,
    })
    print(f"Evaluation {eval_id}/{N_EVALUATIONS}: n={n:.9f} | Starting Abaqus...", flush=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as fh:
        completed = subprocess.run(
            [ABAQUS_CMD, "cae", f"noGUI={WORKER_SCRIPT}"],
            cwd=str(run_dir), env=env, stdout=fh, stderr=subprocess.STDOUT,
            check=False,
        )
    if not result_path.is_file():
        raise RuntimeError(f"Missing result JSON. Check {log_path}")
    with result_path.open("r", encoding="utf-8") as fh:
        result = json.load(fh)
    if completed.returncode != 0 or result.get("status") != "ok":
        raise RuntimeError(
            f"Abaqus failed (exit={completed.returncode}): "
            f"{result.get('error', 'no error message')}. Check {log_path}"
        )
    actual_n = float(result.get("n", float("nan")))
    if not np.isfinite(actual_n) or abs(actual_n - n) > 1e-8:
        raise RuntimeError(f"Worker returned n={actual_n}, requested n={n}")
    error = float(result["nrmse_percent"])
    if not np.isfinite(error) or error < 0:
        raise RuntimeError(f"Invalid NRMSE: {error}")
    print(f"  -> Post-neck NRMSE: {error:.6f}%", flush=True)
    return error


def persist(records, run_dir):
    frame = pd.DataFrame(records)
    frame.to_csv(run_dir / "history.csv", index=False, float_format="%.12g")
    frame.to_excel(run_dir / "Optimization_Results_GridSearch.xlsx", index=False)


def main():
    if N_EVALUATIONS < 2 or not (N_MIN < N_MAX):
        raise ValueError("Grid settings invalid")
    for required in (WORKER_SCRIPT, WORKDIR / "UT_YXZ_M.inp",
                     WORKDIR / "Tensile_test.xls", WORKDIR / "test.xlsx"):
        if not required.is_file():
            raise FileNotFoundError(f"Required input not found: {required}")
    if not Path(ABAQUS_CMD).is_file():
        raise FileNotFoundError(f"Abaqus launcher not found: {ABAQUS_CMD}")

    run_dir = new_run_dir()
    print("Run directory:", run_dir, flush=True)
    config = {
        "algorithm": "Uniform deterministic grid search",
        "single_test": SINGLE_TEST,
        "test_n": TEST_N,
        "grid_bounds": [N_MIN, N_MAX],
        "n_evaluations": N_EVALUATIONS,
        "hardening_points": 40,
        "pre_neck_points": 20,
        "experimental_sheet": "Results",
        "objective": "V9 post-neck NRMSE (%) on 125 PCHIP comparison points",
        "force_and_displacement": "V9 full-specimen reconstruction",
        "identical_physics_to": "Bayesian V9 (worker copied, run naming only)",
    }
    (run_dir / "configuration.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    grid = np.array([TEST_N], dtype=float) if SINGLE_TEST else np.linspace(
        N_MIN, N_MAX, N_EVALUATIONS, dtype=float
    )
    if len(np.unique(grid)) != len(grid):
        raise RuntimeError("Grid has duplicate values")
    records = []
    best_error = float("inf")
    best_n = None
    for eval_id, value in enumerate(grid, start=1):
        n = float(value)
        try:
            score = run_fe(n, eval_id, run_dir)
        except Exception as exc:
            records.append({
                "evaluation": eval_id, "n": n,
                "PostNeck_NRMSE_percent": np.nan,
                "best_so_far_percent": best_error if np.isfinite(best_error) else np.nan,
                "best_n_so_far": best_n,
                "status": "failed", "error": str(exc),
                "abaqus_submissions": eval_id,
            })
            persist(records, run_dir)
            raise
        if score < best_error:
            best_error, best_n = score, n
        records.append({
            "evaluation": eval_id, "n": n,
            "PostNeck_NRMSE_percent": score,
            "best_so_far_percent": best_error,
            "best_n_so_far": best_n,
            "status": "ok", "error": "",
            "abaqus_submissions": eval_id,
        })
        persist(records, run_dir)

    result = {
        "best_n": best_n,
        "best_nrmse_percent": best_error,
        "successful_abaqus_submissions": len(records),
        "unique_n_values": len(set(r["n"] for r in records)),
        "run_id": run_dir.name,
        "single_test": SINGLE_TEST,
    }
    (run_dir / "best_result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(f"Completed: {len(records)} successful FE evaluations", flush=True)
    print(f"Best n={best_n:.9f}; NRMSE={best_error:.6f}%", flush=True)
    print("Results:", run_dir, flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("STOPPED:", exc, file=sys.stderr, flush=True)
        sys.exit(1)
