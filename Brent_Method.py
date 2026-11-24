"""Bounded Brent's method: AISI 316L post-neck inverse calibration.

Uses the physical model and objective from Bayesian V9, unchanged.
Run using ordinary Windows Python: python Brent_Method.py
Requires scipy, numpy, pandas, openpyxl. Abaqus/CAE 2024 runs the worker.

IMPORTANT: scipy.optimize.minimize_scalar(method='bounded') is a bounded
Brent-style LOCAL scalar optimizer, not proof of a global minimum.
Each objective evaluation is one fresh Abaqus analysis; never silently cache.
"""

from datetime import datetime
from pathlib import Path
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

# --------- USER SETTINGS ---------
WORKDIR = Path(r"E:\PC\ABAQUS_SIM\R_D\COMTES_AM\Simulation\Paper\Drawbacks")
ABAQUS_CMD = r"C:\SIMULIA\Commands\abq2024.bat"
WORKER_SCRIPT = WORKDIR / "Brent_worker.py"
SINGLE_TEST = False  # Change to False AFTER n=1.0 test succeeds
TEST_N = 1.0
N_BOUNDS = (0.60, 1.40)
MAX_FE_EVALUATIONS = 15
XATOL = 1.0e-5  # Stop when interval has converged sufficiently
# ---------------------------------


class BudgetExhausted(Exception):
    """Explicitly stop before submitting more than the FE budget."""


def create_run_dir():
    root = WORKDIR / "Brent_runs"
    root.mkdir(parents=True, exist_ok=True)
    stem = datetime.now().strftime("BRENT_%Y%m%d_%H%M%S")
    for counter in range(1000):
        directory = root / (stem if counter == 0 else f"{stem}_{counter:03d}")
        try:
            directory.mkdir()
            return directory
        except FileExistsError:
            pass
    raise RuntimeError("Could not allocate a unique run directory")


def evaluate_fe(n, evaluation, run_dir):
    result_file = run_dir / f"objective_{evaluation:04d}.json"
    log_file = run_dir / f"abaqus_launch_{evaluation:04d}.log"
    if result_file.exists():
        raise RuntimeError(f"Refusing stale result: {result_file}")
    env = os.environ.copy()
    env.update({"BRENT_N": repr(float(n)),
                "BRENT_EVAL_ID": str(evaluation),
                "BRENT_RUN_ID": run_dir.name})
    print(f"Evaluation {evaluation}/{MAX_FE_EVALUATIONS}: n={n:.9f} | Starting Abaqus...", flush=True)
    with log_file.open("w", encoding="utf-8", errors="replace") as logfile:
        process = subprocess.run(
            [ABAQUS_CMD, "cae", f"noGUI={WORKER_SCRIPT}"],
            cwd=str(run_dir), env=env, stdout=logfile,
            stderr=subprocess.STDOUT, check=False,
        )
    if not result_file.is_file():
        raise RuntimeError(f"Missing worker result: {result_file}; see {log_file}")
    result = json.loads(result_file.read_text(encoding="utf-8"))
    if process.returncode != 0 or result.get("status") != "ok":
        raise RuntimeError(
            f"FE failed: {result.get('error', 'unknown failure')} "
            f"(exit code {process.returncode}). See {log_file}"
        )
    returned_n = float(result.get("n", float("nan")))
    if not np.isfinite(returned_n) or abs(returned_n - n) > 1.e-8:
        raise RuntimeError(f"Worker evaluated n={returned_n}, requested n={n}")
    score = float(result["nrmse_percent"])
    if not np.isfinite(score) or score < 0:
        raise RuntimeError(f"Invalid worker NRMSE: {score}")
    print(f"  -> Post-neck NRMSE: {score:.6f}%", flush=True)
    return score


def save_history(records, run_dir):
    table = pd.DataFrame(records)
    table.to_csv(run_dir / "history.csv", index=False, float_format="%.12g")
    table.to_excel(run_dir / "Optimization_Results_Brent.xlsx", index=False)


def main():
    low, high = N_BOUNDS
    if not (0 < low < high) or MAX_FE_EVALUATIONS < 3 or XATOL <= 0:
        raise ValueError("Invalid optimization settings")
    required_files = (WORKER_SCRIPT, WORKDIR / "UT_YXZ_M.inp",
                      WORKDIR / "Tensile_test.xls", WORKDIR / "test.xlsx",
                      Path(ABAQUS_CMD))
    for file in required_files:
        if not file.is_file():
            raise FileNotFoundError(f"Required file missing: {file}")
    run_dir = create_run_dir()
    print("Run directory:", run_dir, flush=True)
    config = {
        "algorithm": "bounded Brent scalar optimization (SciPy minimize_scalar)",
        "single_test": SINGLE_TEST, "test_n": TEST_N,
        "bounds": [low, high], "max_fe_evaluations": MAX_FE_EVALUATIONS,
        "xatol": XATOL,
        "hardening_points": 40, "pre_neck_points": 20,
        "objective": "Bayesian V9 fixed-interval post-neck NRMSE (%) using 125 PCHIP points",
        "experimental_sheet": "Results",
        "worker": "Brent_worker.py (Grid Search/Bayesian V9 mechanics, naming changes only)",
        "note": "Bounded local search; endpoint values not guaranteed to be evaluated."
    }
    (run_dir / "configuration.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8")

    history = []
    best = {"n": None, "error": float("inf")}

    def objective(n):
        n = float(n)
        if len(history) >= MAX_FE_EVALUATIONS:
            raise BudgetExhausted("FE evaluation cap reached")
        evaluation = len(history) + 1
        try:
            error = evaluate_fe(n, evaluation, run_dir)
        except Exception as exc:
            history.append({"evaluation": evaluation, "n": n,
                            "PostNeck_NRMSE_percent": np.nan,
                            "best_so_far_percent": best["error"] if np.isfinite(best["error"]) else np.nan,
                            "best_n_so_far": best["n"],
                            "status": "failed", "error": str(exc),
                            "abaqus_submissions": evaluation})
            save_history(history, run_dir)
            raise
        if error < best["error"]:
            best.update({"n": n, "error": error})
        history.append({"evaluation": evaluation, "n": n,
                        "PostNeck_NRMSE_percent": error,
                        "best_so_far_percent": best["error"],
                        "best_n_so_far": best["n"],
                        "status": "ok", "error": "",
                        "abaqus_submissions": evaluation})
        save_history(history, run_dir)
        return error

    converged = False
    message = ""
    if SINGLE_TEST:
        if not low <= TEST_N <= high:
            raise ValueError("TEST_N outside bounds")
        objective(TEST_N)
        message = "Single FE test only"
    else:
        try:
            result = minimize_scalar(
                objective, method="bounded", bounds=(low, high),
                options={"xatol": XATOL, "maxiter": MAX_FE_EVALUATIONS},
            )
            converged = bool(result.success)
            message = str(result.message)
        except BudgetExhausted:
            message = "Stopped at strict FE evaluation budget"

    summary = {
        "algorithm": "bounded Brent", "best_n": best["n"],
        "best_nrmse_percent": best["error"], "converged": converged,
        "solver_message": message,
        "successful_fe_evaluations": sum(r["status"] == "ok" for r in history),
        "abaqus_submissions": len(history),
        "unique_n_values": len({r["n"] for r in history}),
        "max_fe_evaluations": MAX_FE_EVALUATIONS,
        "single_test": SINGLE_TEST, "run_id": run_dir.name,
    }
    (run_dir / "best_result.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Completed: {len(history)} FE evaluation(s); optimizer converged: {converged}", flush=True)
    print(f"Best n={best['n']:.9f}; Post-neck NRMSE={best['error']:.6f}%", flush=True)
    print("Solver message:", message, flush=True)
    print("Output directory:", run_dir, flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("STOPPED:", exc, file=sys.stderr, flush=True)
        sys.exit(1)
