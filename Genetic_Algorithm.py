"""Elitist real-coded GA with exactly 15 independent ABAQUS FE evaluations.

Use ordinary CPython in CMD, not Abaqus Python.
Requires: numpy, pandas, openpyxl.

IMPORTANT: Identical material model and post-neck NRMSE to Bayesian V9.
Each candidate uses a fresh ABAQUS job; no reused/penalized failed evaluations.

Three generations: generation 0 -> 5 random individuals;
generations 1 and 2 -> 5 new offspring each, with elitist survivor selection.
Total = 5 + 5 + 5 = 15 actual FE evaluations.
"""
from pathlib import Path
from datetime import datetime
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

WORKDIR = Path(r"E:\PC\ABAQUS_SIM\R_D\COMTES_AM\Simulation\Paper\Drawbacks")
ABAQUS_CMD = r"C:\SIMULIA\Commands\abq2024.bat"
WORKER_SCRIPT = WORKDIR / "GA_worker.py"

# ===== USER SETTINGS =====
SINGLE_TEST = False          # Run n=1.0 first. Change to False for full GA.
TEST_N = 1.0
N_BOUNDS = (0.60, 1.40)
FE_BUDGET = 15
POP_SIZE = 5
GENERATIONS = 3           # includes initial generation; 5*3=15 new FE calls
RANDOM_SEED = 42
TOURNAMENT_SIZE = 3
CROSSOVER_PROB = 0.90
BLX_ALPHA = 0.50
MUTATION_PROB = 0.30
MUTATION_SIGMA = 0.08      # absolute units of n
# =========================


def create_run_dir():
    root = WORKDIR / "GA_runs"
    root.mkdir(exist_ok=True)
    prefix = datetime.now().strftime("GA_%Y%m%d_%H%M%S")
    for suffix in range(1000):
        name = prefix if suffix == 0 else f"{prefix}_{suffix:03d}"
        directory = root / name
        try:
            directory.mkdir()
            return name, directory
        except FileExistsError:
            pass
    raise RuntimeError("Could not allocate unique GA run directory")


def call_worker(n, evaluation, run_id, run_dir):
    env = os.environ.copy()
    env["GA_N"] = repr(float(n))
    env["GA_EVAL_ID"] = str(evaluation)
    env["GA_RUN_ID"] = run_id
    log = run_dir / f"abaqus_launch_{evaluation:04d}.log"
    objective_json = run_dir / f"objective_{evaluation:04d}.json"
    print(f"Evaluation {evaluation}/{FE_BUDGET}: n={n:.8f} | Starting Abaqus...", flush=True)
    with log.open("w", encoding="utf-8", errors="replace") as stream:
        proc = subprocess.run(
            [ABAQUS_CMD, "cae", f"noGUI={WORKER_SCRIPT}"],
            cwd=str(run_dir), env=env, stdout=stream,
            stderr=subprocess.STDOUT, check=False,
        )
    if not objective_json.is_file():
        raise RuntimeError(f"Worker did not write objective JSON. Check {log}")
    with objective_json.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    if proc.returncode != 0 or data.get("status") != "ok":
        raise RuntimeError(
            f"FE evaluation {evaluation} FAILED; exit={proc.returncode}; "
            f"detail={data.get('error', 'see worker log')}. Check {log}"
        )
    if abs(float(data.get("n", float("nan"))) - n) > 1e-8:
        raise RuntimeError("Worker output n does not match requested candidate")
    value = float(data["nrmse_percent"])
    if not np.isfinite(value) or value < 0:
        raise RuntimeError(f"Invalid NRMSE returned by worker: {value}")
    print(f"  -> Post-neck NRMSE: {value:.6f}%", flush=True)
    return value


def tournament(population, rng):
    """Select lowest-cost individual among randomly sampled population entries."""
    indices = rng.integers(0, len(population), size=TOURNAMENT_SIZE)
    return min((population[int(j)] for j in indices), key=lambda entry: entry[1])[0]


def make_offspring(population, rng):
    low, high = N_BOUNDS
    p1, p2 = tournament(population, rng), tournament(population, rng)
    if rng.random() < CROSSOVER_PROB:
        spread = abs(p2 - p1)
        child = rng.uniform(min(p1, p2) - BLX_ALPHA * spread,
                            max(p1, p2) + BLX_ALPHA * spread)
    else:
        child = p1 if rng.random() < 0.5 else p2
    if rng.random() < MUTATION_PROB:
        child += rng.normal(loc=0.0, scale=MUTATION_SIGMA)
    # Bound repair; document boundary clipping in manuscript.
    return float(np.clip(child, low, high))


def ga_search(evaluate, rng):
    """Return best individual; evaluate receives (n, generation) and returns loss."""
    if POP_SIZE < 3 or GENERATIONS * POP_SIZE != FE_BUDGET:
        raise ValueError("Choose POP_SIZE >=3 and POP_SIZE*GENERATIONS == FE_BUDGET")
    lo, hi = N_BOUNDS
    # Initial random population: exactly 5 FE calls.
    population = []
    for n in rng.uniform(lo, hi, size=POP_SIZE):
        score = evaluate(float(n), 0)
        population.append((float(n), float(score)))
    print(f"Generation 0 best NRMSE={min(x[1] for x in population):.6f}%", flush=True)
    for generation in range(1, GENERATIONS):
        offspring = []
        for _ in range(POP_SIZE):
            n = make_offspring(population, rng)
            score = evaluate(n, generation)
            offspring.append((n, float(score)))
        # (mu + lambda) elitism: retain best 5 from 5 parents + 5 evaluated children;
        # retained parents do NOT consume new FE evaluations.
        population = sorted(population + offspring, key=lambda entry: entry[1])[:POP_SIZE]
        print(f"Generation {generation} best NRMSE={population[0][1]:.6f}%", flush=True)
    return min(population, key=lambda entry: entry[1])


def main():
    for required in (WORKER_SCRIPT, WORKDIR / "UT_YXZ_M.inp",
                     WORKDIR / "Tensile_test.xls", WORKDIR / "test.xlsx"):
        if not required.is_file():
            raise FileNotFoundError(required)
    if not (0 < N_BOUNDS[0] < N_BOUNDS[1]):
        raise ValueError("Invalid N_BOUNDS")
    if FE_BUDGET != POP_SIZE * GENERATIONS:
        raise ValueError("FE_BUDGET != POP_SIZE*GENERATIONS")
    if not (0 <= CROSSOVER_PROB <= 1 and 0 <= MUTATION_PROB <= 1):
        raise ValueError("Invalid crossover or mutation probability")

    run_id, run_dir = create_run_dir()
    print(f"Run directory: {run_dir}")
    config = {
        "algorithm": "Real-coded GA; tournament, BLX-alpha, Gaussian mutation, (mu+lambda) elitism",
        "single_test": SINGLE_TEST, "test_n": TEST_N, "bounds": N_BOUNDS,
        "fe_budget": FE_BUDGET, "population_size": POP_SIZE,
        "generations_including_initial": GENERATIONS, "seed": RANDOM_SEED,
        "tournament_size": TOURNAMENT_SIZE, "crossover_probability": CROSSOVER_PROB,
        "blx_alpha": BLX_ALPHA, "mutation_probability": MUTATION_PROB,
        "mutation_sigma_n": MUTATION_SIGMA,
        "hardening_points": 40, "experimental_sheet": "Results",
        "objective": "V9 post-neck NRMSE (%), 125 displacement-grid points",
    }
    (run_dir / "configuration.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    history = []

    def evaluate(n, generation):
        idx = len(history) + 1
        n = float(n)
        if not (N_BOUNDS[0] <= n <= N_BOUNDS[1]):
            raise ValueError(f"Out-of-bounds n={n}")
        try:
            score = call_worker(n, idx, run_id, run_dir)
        except Exception as exc:
            history.append({"evaluation": idx, "generation": generation,
                            "n": n, "PostNeck_NRMSE_percent": np.nan,
                            "best_so_far_percent": np.nan,
                            "status": "failed", "error": str(exc)})
            pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
            raise  # Never replace failed physics runs with fake high errors.
        prior = [r["PostNeck_NRMSE_percent"] for r in history if r["status"] == "ok"]
        best_so_far = min(prior + [score])
        history.append({"evaluation": idx, "generation": generation,
                        "n": n, "PostNeck_NRMSE_percent": score,
                        "best_so_far_percent": best_so_far, "status": "ok"})
        df = pd.DataFrame(history)
        df.to_csv(run_dir / "history.csv", index=False)
        df.to_excel(run_dir / "Optimization_Results_GA.xlsx", index=False)
        return score

    if SINGLE_TEST:
        score = evaluate(TEST_N, 0)
        print(f"Single FE test completed. NRMSE={score:.6f}%")
        return
    best_n, best_score = ga_search(evaluate, np.random.default_rng(RANDOM_SEED))
    assert len(history) == FE_BUDGET and all(row["status"] == "ok" for row in history)
    best = {"best_n": best_n, "best_nrmse_percent": best_score,
            "successful_evaluations": len(history), "run_id": run_id}
    (run_dir / "best_result.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    print(f"Best n={best_n:.8f}; Post-neck NRMSE={best_score:.6f}%")
    print(f"Results: {run_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"STOPPED: {exc}", file=sys.stderr)
        sys.exit(1)
