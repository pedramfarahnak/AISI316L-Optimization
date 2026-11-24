"""Hyperopt Tree-structured Parzen Estimator calibration, matching Bayesian V9.

Run using ordinary Windows Python (not Abaqus Python):
    python -m pip install numpy pandas openpyxl hyperopt
    python Hyperopt_TPE.py

IMPORTANT: keep the paired TPE_worker.py alongside this file.
The worker is a parameter-transport/path-renamed copy of Bayesian V9's worker;
its 40-point hardening law, FE extraction and post-neck NRMSE are unchanged.
"""

from datetime import datetime
from functools import partial
from pathlib import Path
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from hyperopt import fmin, hp, tpe, Trials, STATUS_OK

# ============ USER SETTINGS ============
WORKDIR = Path(r"E:\PC\ABAQUS_SIM\R_D\COMTES_AM\Simulation\Paper\Drawbacks")
ABAQUS_CMD = r"C:\SIMULIA\Commands\abq2024.bat"
WORKER_SCRIPT = WORKDIR / "TPE_worker.py"

SINGLE_TEST = False    # True: one FE test; False: launch all 15 TPE evaluations
TEST_N = 1.0
MAX_EVALS = 15
N_INITIAL_POINTS = 3  # CRITICAL: Hyperopt defaults to 20 random startup jobs!
RANDOM_SEED = 42
N_BOUNDS = (0.60, 1.40)
TPE_GAMMA = 0.15      # Same quantile parameter specified in the original article
N_EI_CANDIDATES = 24
# =======================================


def make_run_directory():
    root = WORKDIR / "TPE_runs"
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("TPE_%Y%m%d_%H%M%S")
    for suffix in range(1000):
        run_id = stamp if suffix == 0 else "{}_{}".format(stamp, suffix)
        run_dir = root / run_id
        try:
            run_dir.mkdir()
            return run_id, run_dir
        except FileExistsError:
            pass
    raise RuntimeError("Could not allocate unique TPE run directory")


def evaluate_fe(n, evaluation, run_id, run_dir):
    env = os.environ.copy()
    env["TPE_N"] = repr(float(n))
    env["TPE_EVAL_ID"] = str(int(evaluation))
    env["TPE_RUN_ID"] = run_id

    log = run_dir / "abaqus_launch_{:04d}.log".format(evaluation)
    output = run_dir / "objective_{:04d}.json".format(evaluation)
    command = [ABAQUS_CMD, "cae", "noGUI={}".format(WORKER_SCRIPT)]
    print("Evaluation {}: n={:.8f} | Starting Abaqus...".format(evaluation, n), flush=True)
    with log.open("w", encoding="utf-8", errors="replace") as stream:
        execution = subprocess.run(command, cwd=str(run_dir), env=env,
                                   stdout=stream, stderr=subprocess.STDOUT,
                                   check=False)

    if not output.is_file():
        raise RuntimeError("Worker did not generate {}. See {}".format(output.name, log))
    with output.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    if execution.returncode != 0 or data.get("status") != "ok":
        raise RuntimeError("Abaqus evaluation {} failed (process exit={}, status={}): {}. "
                           "See {} and worker_error_{:04d}.log".format(
                               evaluation, execution.returncode, data.get("status"),
                               data.get("error", "unknown failure"), log, evaluation))
    cost = float(data["nrmse_percent"])
    if not np.isfinite(cost) or cost < 0:
        raise ValueError("Invalid post-neck NRMSE: {}".format(cost))
    print("  -> Post-neck NRMSE: {:.6f}%".format(cost), flush=True)
    return cost


def main():
    for path in [WORKER_SCRIPT, WORKDIR / "UT_YXZ_M.inp",
                 WORKDIR / "Tensile_test.xls", WORKDIR / "test.xlsx"]:
        if not path.is_file():
            raise FileNotFoundError("Required file does not exist: {}".format(path))
    if not Path(ABAQUS_CMD).is_file():
        raise FileNotFoundError("Abaqus executable not found: {}".format(ABAQUS_CMD))
    if not (0 < N_BOUNDS[0] < N_BOUNDS[1]):
        raise ValueError("Expected strictly positive n bounds in ascending order")
    if not (0 < N_INITIAL_POINTS < MAX_EVALS):
        raise ValueError("N_INITIAL_POINTS must be between 1 and MAX_EVALS-1")
    if not (0 < TPE_GAMMA < 1):
        raise ValueError("TPE_GAMMA must be between 0 and 1")

    run_id, run_dir = make_run_directory()
    print("Run directory: {}".format(run_dir), flush=True)
    config = {
        "algorithm": "Hyperopt TPE", "single_test": SINGLE_TEST,
        "test_n": TEST_N, "n_bounds": N_BOUNDS,
        "max_evals": MAX_EVALS, "n_initial_points": N_INITIAL_POINTS,
        "random_seed": RANDOM_SEED, "gamma": TPE_GAMMA,
        "n_ei_candidates": N_EI_CANDIDATES,
        "hardening": "same 40-point formulation as Bayesian V9",
        "nrmse": "fixed post-neck interval, PCHIP, 125 equally-spaced displacements",
        "worker": str(WORKER_SCRIPT),
    }
    (run_dir / "configuration.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    history = []

    def objective(params):
        n = float(params["n"])
        index = len(history) + 1
        try:
            loss = evaluate_fe(n, index, run_id, run_dir)
        except Exception as error:
            history.append({"evaluation": index, "n": n,
                            "PostNeck_NRMSE_percent": np.nan,
                            "status": "failed", "error": str(error)})
            pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
            # Deliberately STOP: a failed solver is not a valid physical measurement.
            raise
        history.append({"evaluation": index, "n": n,
                        "PostNeck_NRMSE_percent": loss, "status": "ok"})
        frame = pd.DataFrame(history)
        frame.to_csv(run_dir / "history.csv", index=False)
        frame.to_excel(run_dir / "Optimization_Results_TPE.xlsx", index=False)
        return {"loss": loss, "status": STATUS_OK}

    if SINGLE_TEST:
        result = objective({"n": TEST_N})
        print("Single FE test completed. NRMSE={:.6f}%".format(result["loss"]))
        return

    # Hyperopt's DEFAULT n_startup_jobs=20; with 15 evaluations that means
    # NO adaptive TPE samples. Explicitly set this to 3 for a fair 15-call test.
    algorithm = partial(tpe.suggest, n_startup_jobs=N_INITIAL_POINTS,
                        gamma=TPE_GAMMA, n_EI_candidates=N_EI_CANDIDATES)
    trials = Trials()
    result = fmin(fn=objective,
                  space={"n": hp.uniform("n", N_BOUNDS[0], N_BOUNDS[1])},
                  algo=algorithm,
                  max_evals=MAX_EVALS,
                  trials=trials,
                  rstate=np.random.default_rng(RANDOM_SEED),
                  show_progressbar=False)

    successful = [item for item in history if item["status"] == "ok"]
    best = min(successful, key=lambda x: x["PostNeck_NRMSE_percent"])
    running_best = float("inf")
    convergence = []
    for item in successful:
        running_best = min(running_best, item["PostNeck_NRMSE_percent"])
        convergence.append({"evaluation": item["evaluation"],
                            "n": item["n"],
                            "NRMSE_percent": item["PostNeck_NRMSE_percent"],
                            "best_so_far_percent": running_best})
    pd.DataFrame(convergence).to_csv(run_dir / "TPE_convergence.csv", index=False)

    summary = {"best_n": best["n"],
               "best_nrmse_percent": best["PostNeck_NRMSE_percent"],
               "best_evaluation": best["evaluation"],
               "successful_evaluations": len(successful),
               "hyperopt_best_n": float(result["n"]),
               "run_id": run_id, "seed": RANDOM_SEED}
    (run_dir / "best_result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("TPE complete: best n={:.8f}; post-neck NRMSE={:.6f}% "
          "at evaluation {}".format(best["n"], best["PostNeck_NRMSE_percent"],
                                    best["evaluation"]))
    print("Results saved in: {}".format(run_dir))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("TPE STOPPED: {}".format(exc), file=sys.stderr)
        sys.exit(1)
