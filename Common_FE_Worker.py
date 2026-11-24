"""Abaqus CAE worker: generate 40-point plasticity table and run FE.

EDIT THE CONFIGURATION BELOW (especially YIELD_STRESS_MPA).
"""
import os
import sys
import traceback
import numpy as np
import pandas as pd
import xlrd
from scipy.interpolate import PchipInterpolator
from abaqus import mdb, session
from abaqusConstants import OFF
import json
import re

# ================= USER SETTINGS - EDIT HERE =================
YIELD_STRESS_MPA = 330.0  # CHANGE THIS to the required TRUE stress threshold in MPa
E_MPA = 220000.0          # Young modulus in MPa
TOTAL_CURVE_POINTS = 40  # TOTAL rows supplied to Abaqus *Plastic
PRE_NECK_POINTS = 20     # Number of pre-neck rows, including join point
EPS_PLASTIC_MAX = 1.0
DIAGNOSTIC_ONLY = False  # Not used in V9; worker always submits a new FE job
# ============================================================

WORKDIR = r"E:\PC\ABAQUS_SIM\R_D\COMTES_AM\Simulation\Paper\Drawbacks"
# Populated for each evaluation by launch(). All material/objective constants and
# numerical functions below are unchanged from the successful Bayesian V9 worker.
RUN_ID = None
RUN_DIR = None
JOB_BASE = None
TEST_XLS = os.path.join(WORKDIR, "test.xlsx")
TENSILE_XLS = os.path.join(WORKDIR, "Tensile_test.xls")
INPUT_FILE = os.path.join(WORKDIR, "UT_YXZ_M.inp")
JOB_BASE = "UT_YXZ_M_BO"
SHEET_EXPERIMENT = "Results"
SHEET_TENSILE = "Tensile_test"
FAILURE_DISP = 3.15509079221417
GRID_POINTS = 125
DISP_END_TOL = 0.01  # mm, full-specimen displacement tolerance
NODE_LABEL_U = 9
NODE_LABEL_RF = 9
PART_INSTANCE = "UT_YXZ_M"  # Verify against odb.rootAssembly.instances keys
DISP_COMPONENT = 1            # U2
FORCE_COMPONENT = 1           # RF2
DISP_FACTOR = 2.0             # Verify quarter model kinematics
FORCE_FACTOR = 2.0            # Verify reaction extraction and symmetry scale
PENALTY = 1e6
NECK_FIT_FRACTION = 0.15  # Original INTERVAL_DER
POINT_LAYOUT = 100.0  # Controls graded sampling, as in original code
N_MIN = 0.05
N_MAX = 1.5
# True: reopen existing ODB; do not regenerate INP or submit an FE job.


def read_experiment():
    df = pd.read_excel(TEST_XLS, sheet_name=SHEET_EXPERIMENT, header=1, usecols="A:B")
    pairs = df.iloc[:, :2].copy()
    pairs.columns = ["u", "f"]
    pairs = pairs.apply(pd.to_numeric, errors="coerce").dropna()
    if len(pairs) < 3:
        raise ValueError("Too few experimental records")
    pairs = pairs.sort_values("u").groupby("u", as_index=False).mean()
    u, force = pairs["u"].to_numpy(), pairs["f"].to_numpy()
    if np.any(np.diff(u) <= 0):
        raise ValueError("Experimental displacement is not strictly increasing")
    if u[0] < -0.1 or u[-1] > 10.0:
        raise ValueError("Unexpected experimental displacement range; verify Results sheet")
    print("Experimental: {} samples, u=[{:.6f},{:.6f}] mm, peak at {:.6f} mm".format(
        len(u), u[0], u[-1], u[int(np.argmax(force))]))
    return u, force


def _isotonic_non_decreasing(values):
    """Pool Adjacent Violators Algorithm, least-squares monotonic stress fit."""
    blocks = []
    for value in values:
        blocks.append([float(value), 1])
        while len(blocks) >= 2 and blocks[-2][0] > blocks[-1][0]:
            m2, c2 = blocks.pop()
            m1, c1 = blocks.pop()
            count = c1 + c2
            blocks.append([(m1 * c1 + m2 * c2) / count, count])
    return np.concatenate([np.full(count, mean) for mean, count in blocks])


def read_pre_neck_flow_curve():
    """Convert paired engineering observations to true stress/plastic strain.

    The source follows the older script: column A = engineering stress MPa,
    column C = extensometer engineering strain in percent, first two rows header.
    """
    workbook = xlrd.open_workbook(TENSILE_XLS)
    worksheet = workbook.sheet_by_name(SHEET_TENSILE)
    paired = []
    for row_idx in range(2, worksheet.nrows):
        row = worksheet.row_values(row_idx)
        if len(row) < 3:
            continue
        try:
            stress, percent = float(row[0]), float(row[2])
        except (TypeError, ValueError):
            continue
        if np.isfinite(stress) and np.isfinite(percent):
            paired.append((stress, percent))
    if len(paired) < 10:
        raise ValueError("Insufficient paired engineering stress/strain rows")

    arr = np.asarray(paired, dtype=float)
    # Preserve original offset convention, but this must be verified against
    # the test preload and zeroing method before reporting final material data.
    stress_eng = arr[:, 0] - arr[0, 0]
    strain_eng = (arr[:, 1] - arr[0, 1]) / 100.0
    if np.any(strain_eng <= -1):
        raise ValueError("Engineering strain cannot be <= -1")
    neck_raw_idx = int(np.argmax(stress_eng))
    if neck_raw_idx < 10:
        raise ValueError("Experimental necking index is too early")
    stress_true = stress_eng * (1.0 + strain_eng)
    strain_true = np.log1p(strain_eng)
    plastic_true = strain_true - stress_true / E_MPA
    eligible = np.flatnonzero(stress_true[:neck_raw_idx + 1] >= YIELD_STRESS_MPA)
    if not len(eligible):
        raise ValueError("Yield threshold {:.3f} MPa not reached before necking".format(YIELD_STRESS_MPA))
    iy = int(eligible[0])
    # Shift plastic-strain origin to selected yield threshold; this is a modeling
    # convention, NOT the absolute plastic strain measured from zero load.
    eps = plastic_true[iy:neck_raw_idx + 1] - plastic_true[iy]
    sig = stress_true[iy:neck_raw_idx + 1]
    # Smooth monotonic strain duplicates/order noise only at tiny tolerance.
    valid = np.isfinite(eps) & np.isfinite(sig)
    eps, sig = eps[valid], sig[valid]
    if len(eps) < 6 or eps[-1] <= 0:
        raise ValueError("Not enough valid pre-neck plastic-strain readings")
    # Keep experimental acquisition order; reject substantial negative strain.
    if np.any(np.diff(eps) < -1e-4):
        raise ValueError("Significant non-monotonic measured plastic strain; inspect extensometer data")
    order = np.argsort(eps, kind='mergesort')
    eps, sig = eps[order], sig[order]
    uniq_eps, inverse = np.unique(eps, return_inverse=True)
    sums = np.bincount(inverse, weights=sig)
    counts = np.bincount(inverse)
    eps, sig = uniq_eps, sums / counts
    if eps[0] != 0.0:
        eps = eps - eps[0]
    smoothed = _isotonic_non_decreasing(sig)
    correction = float(np.max(np.abs(smoothed - sig)))
    if correction > 0.02 * float(np.max(np.abs(sig))):
        raise ValueError("Measured stress requires >2% isotonic adjustment; inspect raw data")
    print("Pre-neck: {} points, eps_neck={:.6g}, sigma_neck={:.6g}, max isotonic correction={:.5g} MPa".format(
        len(eps), eps[-1], smoothed[-1], correction))
    return eps, smoothed


def generate_plastic_curve(n, output_path):
    """Exactly 40 rows with a continuous stress AND tangent at necking.

    Uses original power-law structure sigma = A + B*eps_p**n but corrects
    the coordinate mismatch in original EPS_KRK / EPS_UPRAV, and evaluates
    the derivative using a fit in plastic-strain coordinates.
    """
    if not N_MIN <= n <= N_MAX:
        raise ValueError("Exponent n outside bounds")
    eps, sig = read_pre_neck_flow_curve()
    neck_eps = float(eps[-1])
    neck_stress = float(sig[-1])
    if neck_eps <= 0 or EPS_PLASTIC_MAX <= neck_eps:
        raise ValueError("Invalid neck strain or EPS_PLASTIC_MAX")
    mask = eps >= neck_eps * (1.0 - NECK_FIT_FRACTION)
    if np.count_nonzero(mask) < 5:
        raise ValueError("Too few readings for neck slope estimation")
    # Shift x to neck for improved quadratic-fit conditioning.
    x = eps[mask] - neck_eps
    fit = np.polyfit(x, sig[mask], 2)
    tangent = float(fit[1])
    # A local fit is a derivative estimate, not a smoothing of the entire curve.
    if not np.isfinite(tangent) or tangent <= 0:
        raise ValueError("Neck tangent is nonpositive/invalid: {}".format(tangent))
    B = tangent * neck_eps ** (1.0 - n) / n
    A = neck_stress - B * neck_eps ** n

    if not 3 <= PRE_NECK_POINTS < TOTAL_CURVE_POINTS:
        raise ValueError("Invalid PRE_NECK_POINTS/TOTAL_CURVE_POINTS")
    post_count = TOTAL_CURVE_POINTS - PRE_NECK_POINTS
    # Original-layout inspired graded sampling: more points at low strains
    # and around the transition, but keep the exact necking point explicitly.
    p = np.log(POINT_LAYOUT * TOTAL_CURVE_POINTS) / np.log(TOTAL_CURVE_POINTS)
    t_pre = np.linspace(0.0, 1.0, PRE_NECK_POINTS)
    pre_eps = neck_eps * np.power(t_pre, p)
    pre_eps[-1] = neck_eps
    pre_sig = PchipInterpolator(eps, sig, extrapolate=False)(pre_eps)
    t_post = np.linspace(0.0, 1.0, post_count + 1)[1:]
    post_eps = neck_eps + (EPS_PLASTIC_MAX - neck_eps) * (1 - np.power(1 - t_post, p))
    post_eps[-1] = EPS_PLASTIC_MAX
    post_sig = A + B * np.power(post_eps, n)
    all_eps = np.concatenate((pre_eps, post_eps))
    all_sig = np.concatenate((pre_sig, post_sig))
    if len(all_eps) != TOTAL_CURVE_POINTS:
        raise ValueError("Total curve is not exactly 40 rows")
    if not (np.all(np.isfinite(all_eps)) and np.all(np.isfinite(all_sig))):
        raise ValueError("Nonfinite hardening values")
    if np.any(np.diff(all_eps) <= 0):
        raise ValueError("Plastic strains not strictly increasing")
    if np.any(np.diff(all_sig) < -1e-5):
        raise ValueError("Generated material shows stress softening")
    if abs(pre_sig[-1] - (A + B * neck_eps ** n)) > 1e-6:
        raise ValueError("Stress jump at necking")
    post_tangent = B * n * neck_eps ** (n - 1)
    if not np.isclose(post_tangent, tangent, rtol=1e-9):
        raise ValueError("Tangent jump at necking")

    # Save the EXACT table used by Abaqus, plus an audit file with calculated
    # segment slopes (not a substitute for examining *actual* plasticity).
    with open(output_path, 'w') as handle:
        for stress, strain in zip(all_sig, all_eps):
            handle.write('{:.9f}, {:.9f}\n'.format(stress, strain))
    audit_path = os.path.splitext(output_path)[0] + '_audit.csv'
    with open(audit_path, 'w') as handle:
        handle.write('index,plastic_strain,true_stress_MPa,segment_slope_MPa\n')
        slopes = np.diff(all_sig) / np.diff(all_eps)
        for i in range(len(all_eps)):
            slope_here = slopes[i - 1] if i > 0 else float('nan')
            handle.write('{},{:.12g},{:.12g},{:.12g}\n'.format(
                i + 1, all_eps[i], all_sig[i], slope_here))
    print('Hardening: {} total rows, neck index {}, neck eps {:.6g}, slope {:.6g} MPa'.format(
        len(all_eps), PRE_NECK_POINTS, neck_eps, tangent))
    print('Power law: A={:.6g}, B={:.6g}, n={:.6g}'.format(A, B, n))
    return neck_eps


def update_input_file(source, plastic_path, destination):
    with open(source, "r") as fh:
        lines = fh.readlines()
    # Match keyword exactly; do not accidentally match *Plastic, hardening=...
    indices = [i for i, line in enumerate(lines)
               if re.match(r"^\s*\*Plastic(?:\s|,|$)", line, re.I)]
    if len(indices) != 1:
        raise ValueError("Expected one *Plastic block, found {}".format(len(indices)))
    start = indices[0]
    end = start + 1
    while end < len(lines) and not lines[end].lstrip().startswith("*"):
        end += 1
    with open(plastic_path, "r") as fh:
        material_lines = fh.readlines()
    updated = lines[:start] + [lines[start]] + material_lines + lines[end:]
    with open(destination, "w") as fh:
        fh.writelines(updated)


def simulate(input_path, job):
    if not os.path.isfile(input_path):
        raise FileNotFoundError("Generated INP missing: " + input_path)
    for extension in (".odb", ".sta", ".dat", ".msg", ".lck"):
        old = os.path.join(RUN_DIR, job + extension)
        if os.path.exists(old):
            raise RuntimeError("Refusing to overwrite existing job output: " + old)
    mdb.JobFromInputFile(name=job, inputFileName=os.path.abspath(input_path))
    mdb.jobs[job].submit()
    mdb.jobs[job].waitForCompletion()
    sta_path = os.path.join(RUN_DIR, job + ".sta")
    if not os.path.isfile(sta_path):
        raise RuntimeError("No Abaqus status file: " + sta_path)
    with open(sta_path, "r", errors="replace") as fh:
        sta_text = fh.read().upper()
    if "COMPLETED SUCCESSFULLY" not in sta_text:
        raise RuntimeError("Solver did not complete successfully; inspect .sta/.msg: " + job)
    odb_path = os.path.join(RUN_DIR, job + ".odb")
    if not os.path.isfile(odb_path):
        raise RuntimeError("Solver did not produce ODB: " + odb_path)
    return odb_path


def extract_curve(odb_path):
    odb = session.openOdb(name=odb_path, readOnly=True)
    try:
        step = odb.steps["Step-1"]
        instance = odb.rootAssembly.instances[PART_INSTANCE]
        displacement, reaction = [], []
        for frame in step.frames:
            u_values = frame.fieldOutputs["U"].getSubset(region=instance).values
            rf_values = frame.fieldOutputs["RF"].getSubset(region=instance).values
            u = next(v for v in u_values if v.nodeLabel == NODE_LABEL_U)
            rf = next(v for v in rf_values if v.nodeLabel == NODE_LABEL_RF)
            displacement.append(DISP_FACTOR * float(u.data[DISP_COMPONENT]))
            reaction.append(FORCE_FACTOR * float(rf.data[FORCE_COMPONENT]))
        return np.asarray(displacement), np.asarray(reaction)
    finally:
        odb.close()


def prepare_fe_curve(u_fe, f_fe):
    u_fe = np.asarray(u_fe, dtype=float)
    f_fe = np.asarray(f_fe, dtype=float)
    if len(u_fe) != len(f_fe) or len(u_fe) < 3:
        raise ValueError("Invalid FE displacement-force array lengths")
    if not np.all(np.isfinite(u_fe)) or not np.all(np.isfinite(f_fe)):
        raise ValueError("Nonfinite FE displacement/force data")
    df = pd.DataFrame({"u": u_fe, "f": f_fe})
    df = df.sort_values("u").groupby("u", as_index=False).mean()
    return df["u"].to_numpy(), df["f"].to_numpy()


def post_neck_nrmse(u_fe, f_fe, u_exp, f_exp, neck_disp, odb_path, eval_id):
    u_fe, f_fe = prepare_fe_curve(u_fe, f_fe)
    lower = float(neck_disp)
    upper = float(min(FAILURE_DISP, u_exp[-1]))
    diag = os.path.join(RUN_DIR, "displacement_{:04d}.txt".format(eval_id))
    with open(diag, "w") as fh:
        fh.write("ODB: {}\n".format(odb_path))
        fh.write("FE range: {:.9f} to {:.9f} mm\n".format(u_fe[0], u_fe[-1]))
        fh.write("EXP range: {:.9f} to {:.9f} mm\n".format(u_exp[0], u_exp[-1]))
        fh.write("Comparison: {:.9f} to {:.9f} mm\n".format(lower, upper))
    # Do not silently shrink the interval for one candidate.
    if upper <= lower or u_fe[0] > lower + 1e-6 or u_fe[-1] < upper - DISP_END_TOL:
        raise ValueError("Incomplete fixed post-neck comparison interval; see " + diag)
    if u_fe[-1] < upper:
        # Small round-off/experimental endpoint discrepancy: compare to the
        # fixed endpoint only if bounded within tolerance; no extrapolation.
        upper = float(u_fe[-1])
        # This case is *not* used by default: require full experimental coverage.
        raise ValueError("FE ends short of fixed comparison interval by {:.6g} mm".format(
            min(FAILURE_DISP, u_exp[-1]) - upper))
    grid = np.linspace(lower, upper, GRID_POINTS)
    exp_grid = PchipInterpolator(u_exp, f_exp, extrapolate=False)(grid)
    fe_grid = PchipInterpolator(u_fe, f_fe, extrapolate=False)(grid)
    if not np.all(np.isfinite(exp_grid)) or not np.all(np.isfinite(fe_grid)):
        raise ValueError("Nonfinite interpolated force")
    ref_force = float(np.max(np.abs(f_exp)))
    if ref_force <= 0:
        raise ValueError("Invalid reference force")
    errors = fe_grid - exp_grid
    table = pd.DataFrame({"u_mm": grid, "F_exp_N": exp_grid,
                          "F_FE_N": fe_grid, "error_N": errors})
    table.to_csv(os.path.join(RUN_DIR, "comparison_{:04d}.csv".format(eval_id)), index=False)
    return 100.0 * float(np.sqrt(np.mean(errors ** 2))) / ref_force


def run(n, eval_id):
    os.makedirs(RUN_DIR, exist_ok=True)
    os.chdir(RUN_DIR)
    result_path = os.path.join(RUN_DIR, "objective_{:04d}.json".format(eval_id))
    error_path = os.path.join(RUN_DIR, "worker_error_{:04d}.log".format(eval_id))
    job = "{}_e{:04d}".format(JOB_BASE, eval_id)
    plastic_path = os.path.join(RUN_DIR, "Plastic_Curve_{:04d}.txt".format(eval_id))
    input_path = os.path.join(RUN_DIR, job + ".inp")
    for stale in (result_path, error_path):
        if os.path.exists(stale):
            raise RuntimeError("Output from this evaluation already exists: " + stale)
    record = {"n": float(n), "evaluation": int(eval_id), "status": "failed",
              "nrmse_percent": None, "job": job}
    try:
        for required in (TEST_XLS, TENSILE_XLS, INPUT_FILE):
            if not os.path.isfile(required):
                raise FileNotFoundError("Missing input: " + required)
        u_exp, f_exp = read_experiment()
        neck_disp = float(u_exp[int(np.argmax(f_exp))])
        generate_plastic_curve(n, plastic_path)
        update_input_file(INPUT_FILE, plastic_path, input_path)
        odb_path = simulate(input_path, job)
        u_fe, f_fe = extract_curve(odb_path)
        if not np.isfinite(u_fe[-1]):
            raise ValueError("Invalid final FE displacement")
        value = post_neck_nrmse(u_fe, f_fe, u_exp, f_exp, neck_disp, odb_path, eval_id)
        record["nrmse_percent"] = float(value)
        record["status"] = "ok"
        print("Evaluation {} n={:.7f} Post-neck NRMSE={:.6f}%".format(eval_id, n, value))
    except Exception as exc:
        record["error"] = str(exc)
        with open(error_path, "w") as log:
            log.write(traceback.format_exc())
        print("WORKER FAILED: " + traceback.format_exc())
    with open(result_path, "w") as fp:
        json.dump(record, fp, indent=2)
    if record["status"] != "ok":
        raise RuntimeError("Evaluation failed: " + str(record.get("error")))



# Controller compatibility: unchanged env var names and unchanged output folders.
METHOD_CONFIG = {
    "BO": ("BAYESIAN", "BO_runs", "UT_YXZ_M_BO"),
    "TPE": ("TPE", "TPE_runs", "UT_YXZ_M_TPE"),
    "GA": ("GA", "GA_runs", "UT_YXZ_M_GA"),
    "PSO": ("PSO", "PSO_runs", "UT_YXZ_M_PSO"),
    "GRID": ("GRID", "GridSearch_runs", "UT_YXZ_M_GRID"),
    "BRENT": ("BRENT", "Brent_runs", "UT_YXZ_M_BRENT"),
}


def launch(method):
    """Route legacy Controller variables to exactly the same FE evaluation."""
    global RUN_ID, RUN_DIR, JOB_BASE
    if method not in METHOD_CONFIG:
        raise ValueError("Unsupported optimization method: " + repr(method))
    prefix, folder, job_base = METHOD_CONFIG[method]
    RUN_ID = os.environ.get(prefix + "_RUN_ID", "manual")
    if not re.match(r"^[A-Za-z0-9_-]{1,40}$", RUN_ID):
        raise ValueError("Invalid " + prefix + "_RUN_ID")
    RUN_DIR = os.path.join(WORKDIR, folder, RUN_ID)
    JOB_BASE = job_base
    try:
        n_input = float(os.environ[prefix + "_N"])
        eval_input = int(os.environ[prefix + "_EVAL_ID"])
    except (KeyError, ValueError, TypeError) as exc:
        raise SystemExit("Invalid " + prefix + "_N / " + prefix + "_EVAL_ID: " + str(exc))
    run(n_input, eval_input)
