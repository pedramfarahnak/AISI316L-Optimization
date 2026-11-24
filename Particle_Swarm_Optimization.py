"""Reproducible one-dimensional PSO for Abaqus post-neck inverse calibration.

15 ACTUAL FE attempts: 5 particles x (initial evaluation + 2 updates).
The ABAQUS constitutive model and objective are copied from the successful GA/V9
worker; this script only changes how n values are proposed.

Run in system CPython: python Particle_Swarm_Optimization.py
Dependencies: numpy pandas openpyxl. Abaqus CAE runs the PSO_worker.py separately.
"""
from pathlib import Path
from datetime import datetime
import json
import os
import subprocess
import sys
import numpy as np
import pandas as pd

# ---- USER SETTINGS ----
WORKDIR = Path(r"E:\PC\ABAQUS_SIM\R_D\COMTES_AM\Simulation\Paper\Drawbacks")
ABAQUS_CMD = r"C:\SIMULIA\Commands\abq2024.bat"
WORKER_SCRIPT = WORKDIR / "PSO_worker.py"
SINGLE_TEST = False
TEST_N = 1.0
N_BOUNDS = (0.60, 1.40)
FE_BUDGET = 15
N_PARTICLES = 5
N_ROUNDS = 3       # Includes initial evaluation; total 5 x 3 = 15
RANDOM_SEED = 42
INERTIA = 0.70
COGNITIVE = 1.50  # c1
SOCIAL = 1.50     # c2
VELOCITY_CLAMP_FRACTION = 0.20  # vmax = 0.16 n
AVOID_DUPLICATES = True
MIN_N_SEPARATION = 1e-8
# -----------------------


def create_run_dir():
    root = WORKDIR / "PSO_runs"
    root.mkdir(parents=True, exist_ok=True)
    prefix = datetime.now().strftime("PSO_%Y%m%d_%H%M%S")
    for k in range(1000):
        directory = root / (prefix if k == 0 else f"{prefix}_{k:03d}")
        try:
            directory.mkdir()
            return directory
        except FileExistsError:
            pass
    raise RuntimeError("Unable to create unique PSO run directory")


def call_worker(n, eval_id, run_dir):
    env = os.environ.copy()
    env.update({"PSO_N": repr(float(n)),
                "PSO_EVAL_ID": str(eval_id), "PSO_RUN_ID": run_dir.name})
    log = run_dir / f"abaqus_launch_{eval_id:04d}.log"
    result_file = run_dir / f"objective_{eval_id:04d}.json"
    print(f"Evaluation {eval_id}/{FE_BUDGET}: n={n:.9f} | Starting Abaqus...", flush=True)
    with log.open("w", encoding="utf-8", errors="replace") as fh:
        proc = subprocess.run([ABAQUS_CMD, "cae", f"noGUI={WORKER_SCRIPT}"],
                              cwd=str(run_dir), env=env, stdout=fh,
                              stderr=subprocess.STDOUT, check=False)
    if not result_file.is_file():
        raise RuntimeError(f"No result JSON for evaluation {eval_id}; inspect {log}")
    with result_file.open(encoding="utf-8") as fh:
        result = json.load(fh)
    if proc.returncode != 0 or result.get("status") != "ok":
        raise RuntimeError(f"FE failed (exit={proc.returncode}): "
                           f"{result.get('error', '')}; log: {log}")
    if abs(float(result.get("n", float("nan"))) - n) > 1e-8:
        raise RuntimeError("Mismatch between requested n and returned result")
    score = float(result["nrmse_percent"])
    if not np.isfinite(score) or score < 0:
        raise RuntimeError(f"Invalid objective returned: {score}")
    print(f"  -> Post-neck NRMSE: {score:.6f}%", flush=True)
    return score


def enforce_unique(x, seen, rng):
    """Avoid spending expensive FE calls on exactly repeated parameter values.

    Small nudge is recorded through the actual evaluated n. This is a deviation
    from standard PSO and is disclosed in the saved configuration.
    """
    lo, hi = N_BOUNDS
    x = float(np.clip(x, lo, hi))
    if not AVOID_DUPLICATES:
        return x
    if not any(abs(x - old) <= MIN_N_SEPARATION for old in seen):
        return x
    for step in (1e-5, 1e-4, 1e-3, 1e-2):
        for sign in (1., -1.):
            candidate = float(np.clip(x + sign * step, lo, hi))
            if not any(abs(candidate - old) <= MIN_N_SEPARATION for old in seen):
                return candidate
    for _ in range(100):
        candidate = float(rng.uniform(lo, hi))
        if not any(abs(candidate - old) <= MIN_N_SEPARATION for old in seen):
            return candidate
    raise RuntimeError("Could not generate a unique parameter")


def pso_search(evaluate, rng):
    lo, hi = N_BOUNDS
    vmax = VELOCITY_CLAMP_FRACTION * (hi - lo)
    positions = rng.uniform(lo, hi, size=N_PARTICLES)
    velocities = rng.uniform(-vmax, vmax, size=N_PARTICLES)
    personal_best_pos = positions.copy()
    personal_best_scores = np.full(N_PARTICLES, np.inf)
    global_best_pos = float("nan")
    global_best_score = float("inf")
    seen = []

    for iteration in range(N_ROUNDS):
        # Synchronous swarm update: the global best is frozen while proposing
        # all next-round positions, avoiding ordering bias among particles.
        if iteration > 0:
            r1 = rng.random(N_PARTICLES)
            r2 = rng.random(N_PARTICLES)
            velocities = (INERTIA * velocities
                          + COGNITIVE * r1 * (personal_best_pos - positions)
                          + SOCIAL * r2 * (global_best_pos - positions))
            velocities = np.clip(velocities, -vmax, vmax)
            raw_positions = positions + velocities
            positions = np.clip(raw_positions, lo, hi)
            # Zero velocity when the particle reaches a bound.
            velocities[positions != raw_positions] = 0.0

        round_positions = []
        for i in range(N_PARTICLES):
            n = enforce_unique(positions[i], seen, rng)
            positions[i] = n
            seen.append(n)
            round_positions.append(n)
        # Freeze proposed positions within the generation. Fitness updates occur
        # only after all particles in this generation have been evaluated.
        round_scores = [evaluate(n, iteration, i + 1) for i, n in enumerate(round_positions)]
        for i, score in enumerate(round_scores):
            if score < personal_best_scores[i]:
                personal_best_scores[i] = score
                personal_best_pos[i] = round_positions[i]
            if score < global_best_score:
                global_best_score = score
                global_best_pos = round_positions[i]
        print(f"Round {iteration}: best n={global_best_pos:.9f}; "
              f"NRMSE={global_best_score:.6f}%", flush=True)
    return global_best_pos, global_best_score


def main():
    if N_PARTICLES * N_ROUNDS != FE_BUDGET:
        raise ValueError("FE_BUDGET must equal N_PARTICLES * N_ROUNDS")
    if not (0 < N_BOUNDS[0] < N_BOUNDS[1]):
        raise ValueError("Invalid bounds")
    for required in (WORKER_SCRIPT, WORKDIR / "UT_YXZ_M.inp",
                     WORKDIR / "Tensile_test.xls", WORKDIR / "test.xlsx"):
        if not required.is_file():
            raise FileNotFoundError(required)
    run_dir = create_run_dir()
    print("Run directory:", run_dir, flush=True)
    config = {"algorithm": "Synchronous real-valued PSO", "single_test": SINGLE_TEST,
              "test_n": TEST_N, "bounds": N_BOUNDS, "fe_budget": FE_BUDGET,
              "particles": N_PARTICLES, "rounds_including_initial": N_ROUNDS,
              "random_seed": RANDOM_SEED, "inertia": INERTIA,
              "cognitive": COGNITIVE, "social": SOCIAL,
              "velocity_clamp_fraction": VELOCITY_CLAMP_FRACTION,
              "boundary_handling": "clamp position and zero outward velocity",
              "avoid_duplicates": AVOID_DUPLICATES,
              "minimum_n_separation": MIN_N_SEPARATION,
              "hardening_points": 40, "experimental_sheet": "Results",
              "objective": "V9 post-neck NRMSE (%), 125 comparison points"}
    (run_dir / "configuration.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    records = []

    def evaluate(n, round_id, particle):
        eval_id = len(records) + 1
        try:
            score = call_worker(float(n), eval_id, run_dir)
        except Exception as exc:
            records.append({"evaluation": eval_id, "round": round_id,
                            "particle": particle, "n": n, "PostNeck_NRMSE_percent": np.nan,
                            "best_so_far_percent": np.nan, "status": "failed", "error": str(exc),
                            "abaqus_submissions": eval_id})
            pd.DataFrame(records).to_csv(run_dir / "history.csv", index=False)
            raise
        good = [r["PostNeck_NRMSE_percent"] for r in records if r["status"] == "ok"]
        records.append({"evaluation": eval_id, "round": round_id,
                        "particle": particle, "n": float(n), "PostNeck_NRMSE_percent": score,
                        "best_so_far_percent": min(good + [score]),
                        "status": "ok", "abaqus_submissions": eval_id})
        df = pd.DataFrame(records)
        df.to_csv(run_dir / "history.csv", index=False)
        df.to_excel(run_dir / "Optimization_Results_PSO.xlsx", index=False)
        return score

    if SINGLE_TEST:
        score = evaluate(TEST_N, 0, 1)
        print(f"Single FE test completed. NRMSE={score:.6f}%", flush=True)
        return

    best_n, best_score = pso_search(evaluate, np.random.default_rng(RANDOM_SEED))
    if len(records) != FE_BUDGET or any(r["status"] != "ok" for r in records):
        raise RuntimeError("Completed run did not produce 15 successful evaluations")
    n_unique = len(set(r["n"] for r in records))
    best = {"best_n": float(best_n), "best_nrmse_percent": float(best_score),
            "successful_abaqus_submissions": len(records),
            "unique_n_values": n_unique, "run_id": run_dir.name}
    (run_dir / "best_result.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    print(f"Best n={best_n:.9f}; NRMSE={best_score:.6f}%", flush=True)
    print(f"Successful ABAQUS submissions: {len(records)}; unique n: {n_unique}")
    print("Results:", run_dir, flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("STOPPED:", exc, file=sys.stderr)
        sys.exit(1)
