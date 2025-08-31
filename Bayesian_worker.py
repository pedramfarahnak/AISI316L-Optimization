import sys
import os
import numpy as np
import pandas as pd
from abaqus import mdb, session
import xlrd
from scipy.interpolate import interp1d

# -- CONFIG --
WORKDIR = r"E:\PC\ABAQUS_SIM\R_D\COMTES_AM\Simulation\Bayesian"
TEST_XLS = os.path.join(WORKDIR, "test.xlsx")
TENSILE_XLS = os.path.join(WORKDIR, "Tensile_test.xls")
INPUT_FILE = os.path.join(WORKDIR, "UT_YXZ_M.inp")
JOB_BASE = "UT_YXZ_M"
FAILURE_DISP = 3.15509079221417
POLY_DEGREE = 25
NODE_LABEL_U = 9
NODE_LABEL_RF = 9
PART_INSTANCE = "UT_YXZ_M"
ASSY_INSTANCE = "UT_YXZ_M"

os.chdir(WORKDIR)

def generate_plastic_curve(n, N=20):
    VSTUP = xlrd.open_workbook(TENSILE_XLS)
    LIST1 = VSTUP.sheet_by_name('Tensile_test')
    DATA = [LIST1.row_values(j) for j in range(LIST1.nrows)]
    SIG = [DATA[j][0] - DATA[2][0] for j in range(2, len(DATA))]
    EPS = [(DATA[j][2] - DATA[2][2]) / 100.0 for j in range(2, len(DATA))]
    SIG_SKUT = [SIG[j] * (1 + EPS[j]) for j in range(len(EPS))]
    EPS_LOG = [np.log(1 + EPS[j]) for j in range(len(EPS))]
    i_kluz = 0
    while SIG_SKUT[i_kluz] < 170:
        i_kluz += 1
    EPS_UPRAV = [EPS_LOG[i + i_kluz] - EPS_LOG[i_kluz] for i in range(len(EPS_LOG) - i_kluz)]
    SIG_UPRAV = SIG_SKUT[i_kluz:]
    EPS_KRK = EPS_LOG[SIG.index(max(SIG))]
    i_krk = next(i for i, val in enumerate(EPS_UPRAV) if val >= EPS_KRK)
    i_start = next(i for i, val in enumerate(EPS_UPRAV) if val >= EPS_KRK * (1 - 0.15))
    EPS_der = EPS_UPRAV[i_start:i_krk]
    SIG_der = SIG_UPRAV[i_start:i_krk]
    f1, f2, f3 = np.square(EPS_der), EPS_der, [1] * len(EPS_der)
    A = np.array([f1, f2, f3]).T
    Y = np.array(SIG_der).reshape(-1, 1)
    b = np.linalg.lstsq(A, Y, rcond=None)[0]
    derivace = 2 * b[0] * EPS_KRK + b[1]
    B = (derivace * (EPS_KRK) ** (1 - n)) / n
    A_val = SIG_UPRAV[i_krk] - (EPS_KRK * derivace) / n
    EXPONONT = np.log(100 * N) / np.log(N)
    VAHA = 1 / ((N - 1) ** EXPONONT)
    EPS_INTERP = [VAHA * i ** EXPONONT for i in range(N)]
    INTERP = interp1d(EPS_UPRAV, SIG_UPRAV, fill_value="extrapolate")
    SIG_INTERP = [float(INTERP(eps)) if eps < EPS_KRK else float(A_val + B * eps ** n) for eps in EPS_INTERP]
    with open('Plastic_Curve.txt', 'w') as f:
        for sig, eps in zip(SIG_INTERP, EPS_INTERP):
            f.write(f"{sig:.6f} , {eps:.6f}\n")

def update_input_file(original_file, hardening_file, updated_file):
    with open(original_file, 'r') as file:
        lines = file.readlines()
    start_idx, end_idx = None, None
    for i, line in enumerate(lines):
        if '*Plastic' in line:
            start_idx = i + 1
            for j in range(start_idx, len(lines)):
                if lines[j].startswith('*') and not lines[j].strip().startswith('*Plastic'):
                    end_idx = j
                    break
            if end_idx:
                break
    if start_idx is not None and end_idx is not None:
        lines = lines[:start_idx - 1] + lines[end_idx:]
    with open(hardening_file, 'r') as hf:
        hardening_data = hf.readlines()
    if start_idx is not None:
        lines = lines[:start_idx - 1] + ['*Plastic\n'] + hardening_data + lines[start_idx - 1:]
    with open(updated_file, 'w') as file:
        file.writelines(lines)
    return True

def run_simulation(input_file, job_name):
    mdb.JobFromInputFile(name=job_name, inputFileName=input_file)
    mdb.jobs[job_name].submit()
    mdb.jobs[job_name].waitForCompletion()

def collect_results(job_name):
    odb_path = job_name + ".odb"
    odb = session.openOdb(name=odb_path)
    step = odb.steps["Step-1"]
    disp, force = [], []
    for fr in step.frames:
        u_fld = fr.fieldOutputs["U"].getSubset(region=odb.rootAssembly.instances[PART_INSTANCE])
        r_fld = fr.fieldOutputs["RF"].getSubset(region=odb.rootAssembly.instances[ASSY_INSTANCE])
        du = next(v for v in u_fld.values if v.nodeLabel == NODE_LABEL_U).data[1]
        df = next(v for v in r_fld.values if v.nodeLabel == NODE_LABEL_RF).data[1]
        disp.append(2 * float(du))
        force.append(2 * float(df))
    odb.close()
    return np.array(disp), np.array(force)

def main():
    try:
        # Get n-value from argument
        for arg in sys.argv:
            try:
                n = float(arg)
                break
            except ValueError:
                continue
        else:
            n = 0.5  # fallback value

        # Experimental data and polynomial
        df_exp = pd.read_excel(TEST_XLS, sheet_name="Results")
        u_exp = pd.to_numeric(df_exp.iloc[:, 0], errors="coerce").dropna().to_numpy()
        f_exp = pd.to_numeric(df_exp.iloc[:, 1], errors="coerce").dropna().to_numpy()
        u_min, u_max = u_exp.min(), u_exp.max()
        u_norm = (u_exp - u_min) / (u_max - u_min)
        coeffs = np.polyfit(u_norm, f_exp, 25)
        poly_exp = np.poly1d(coeffs)
        def interpolate_exp(u_query):
            u_query_norm = (u_query - u_min) / (u_max - u_min)
            return poly_exp(u_query_norm)
        # Prepare job
        iter_tag = f"{n:.4f}".replace(".", "_")
        job_name = f"{JOB_BASE}_n{iter_tag}"
        inp_name = f"{job_name}.inp"
        generate_plastic_curve(n)
        update_input_file(INPUT_FILE, 'Plastic_Curve.txt', inp_name)
        run_simulation(inp_name, job_name)
        U_num, F_num = collect_results(job_name)
        mask = U_num <= FAILURE_DISP
        U_tr, F_tr = U_num[mask], F_num[mask]
        F_exp_tr = interpolate_exp(U_tr)
        cost_pct = np.mean(np.abs(F_tr - F_exp_tr) / F_exp_tr) * 100
        with open("objective.txt", "w") as f:
            f.write(str(cost_pct))
        print("Objective (cost function):", cost_pct)
    except Exception as e:
        with open("objective.txt", "w") as f:
            f.write(str(1e6))
        print("Failed in worker:", e)

if __name__ == "__main__":
    main()
