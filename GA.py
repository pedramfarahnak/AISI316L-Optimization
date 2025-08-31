import numpy as np
import pandas as pd
from abaqus import mdb, session
from scipy.interpolate import interp1d
import xlrd
import random
from deap import base, creator, tools, algorithms

# --- Input Parameters ---
input_file = r'E:\PC\ABAQUS_SIM\R_D\COMTES_AM\Simulation\GA\UT_YXZ_M.inp'
working_directory = r'E:\PC\ABAQUS_SIM\R_D\COMTES_AM\Simulation\GA'
job_name_base = 'UT_YXZ_M'
num_generations = 10
population_size = 5
cx_prob = 0.6  # Crossover probability
mut_prob = 0.3 # Mutation probability
node_label_displacement = 9
node_label_force = 9
part_instance_name = 'UT_YXZ_M'
assembly_instance_name = 'UT_YXZ_M'
exp_data_file = r'E:\PC\ABAQUS_SIM\R_D\COMTES_AM\Simulation\GA\test.xlsx'
convergence_threshold = 0.05

# --- Storage ---
n_values = []
cost_values = []
numerical_displacements = []
numerical_forces = []
interpolated_experimental_displacements = []
interpolated_experimental_forces = []
objective_call_counter = 0

# --- Functions (from your previous code, adjust paths as needed) ---
def load_experimental_data(file_path):
    try:
        data = pd.read_excel(file_path, sheet_name='Results')
        displacement = pd.to_numeric(data.iloc[:, 0], errors='coerce').dropna().to_numpy()
        force = pd.to_numeric(data.iloc[:, 1], errors='coerce').dropna().to_numpy()
        print(f"Loaded {len(displacement)} data points from the 'Results' sheet.")
        return displacement, force
    except Exception as e:
        print(f"Error reading data from {file_path}: {e}")
        return None, None

def polynomial_fit(displacement, force, degree=25):
    try:
        displacement_min = np.min(displacement)
        displacement_max = np.max(displacement)
        normalized_displacement = (displacement - displacement_min) / (displacement_max - displacement_min)
        coeffs = np.polyfit(normalized_displacement, force, degree)
        polynomial = np.poly1d(coeffs)
        print(f"Fitted a degree-{degree} polynomial to the data.")
        return polynomial, displacement_min, displacement_max
    except Exception as e:
        print(f"Error during polynomial fitting: {e}")
        return None, None, None

def run_simulation(input_file, job_name):
    print(f"Running Abaqus simulation for job: {job_name}")
    mdb.JobFromInputFile(name=job_name, inputFileName=input_file)
    mdb.jobs[job_name].submit()
    mdb.jobs[job_name].waitForCompletion()
    print(f"Job {job_name} completed successfully.")

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
    print(f"Updated input file created: {updated_file}")
    return True

def collect_results(job_name, node_label_displacement, node_label_force, part_instance_name, assembly_instance_name):
    print(f"Collecting results for job: {job_name}")
    odb = session.openOdb(name=job_name + '.odb')
    step = odb.steps['Step-1']
    frames = step.frames
    displacements = []
    forces = []
    for frame in frames:
        instance = odb.rootAssembly.instances[part_instance_name]
        disp_output = frame.fieldOutputs['U']
        for value in disp_output.getSubset(region=instance).values:
            if value.nodeLabel == node_label_displacement:
                displacements.append(2 * float(value.data[1]))
                break
        assembly_instance = odb.rootAssembly.instances[assembly_instance_name]
        force_output = frame.fieldOutputs['RF']
        for value in force_output.getSubset(region=assembly_instance).values:
            if value.nodeLabel == node_label_force:
                forces.append(2 * float(value.data[1]))
                break
    odb.close()
    numerical_displacements.append(displacements)
    numerical_forces.append(forces)
    return forces, displacements

def compare_results(numerical, experimental):
    F_numerical, U_numerical = numerical
    F_exp, U_exp = experimental
    if U_exp is None or F_exp is None:
        print("Experimental data not loaded properly.")
        return np.inf
    if len(F_numerical) != len(F_exp):
        print("Mismatch in data length.")
        return np.inf
    J_force = np.mean(np.abs(np.array(F_numerical) - np.array(F_exp)) / np.array(F_exp))
    print(f"J_force: {J_force}")
    return J_force

def generate_plastic_curve(n, N=20):
    try:
        VSTUP = xlrd.open_workbook('Tensile_test.xls')
        LIST1 = VSTUP.sheet_by_name('Tensile_test')
        DATA = [LIST1.row_values(j) for j in range(LIST1.nrows)]
        SIG = [DATA[j][0] - DATA[2][0] for j in range(2, len(DATA))]
        EPS = [(DATA[j][2] - DATA[2][2]) / 100.0 for j in range(2, len(DATA))]
    except Exception as e:
        print(f"Error reading 'Tensile_test.xls': {e}")
        raise
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
    f1, f2, f3 = np.square(EPS_der), EPS_der, [1]*len(EPS_der)
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
    print("Plastic_Curve.txt has been generated.")

# --- Main GA Optimization ---

U_exp, F_exp = load_experimental_data(exp_data_file)
poly, displacement_min, displacement_max = polynomial_fit(U_exp, F_exp, degree=25)

def ga_objective(individual):
    n = individual[0]
    global objective_call_counter
    iteration = objective_call_counter + 1
    n_str = f"{n:.4f}".replace('.', '_')
    job_name = f"{job_name_base}_iter{iteration}_Opt_{n_str}"
    updated_input_file = f"{job_name}.inp"
    try:
        generate_plastic_curve(n)
        if not update_input_file(input_file, 'Plastic_Curve.txt', updated_input_file):
            print("Update input file failed.")
            raise RuntimeError("Update input file failed.")
        run_simulation(updated_input_file, job_name)
        F_numerical, U_numerical = collect_results(
            job_name, node_label_displacement, node_label_force, part_instance_name, assembly_instance_name
        )
        U_exp_interp = np.linspace(0, max(U_numerical), len(U_numerical))
        normalized_U_exp = (U_exp_interp - displacement_min) / (displacement_max - displacement_min)
        F_exp_interp = poly(normalized_U_exp)
        interpolated_experimental_displacements.append(U_exp_interp)
        interpolated_experimental_forces.append(F_exp_interp)
        J = compare_results((F_numerical, U_numerical), (F_exp_interp, U_exp_interp))
    except Exception as e:
        print(f"Simulation failed for n={n}: {e}")
        J = np.inf
    n_values.append(n)
    cost_values.append(J)
    objective_call_counter += 1
    return J,

# DEAP setup
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_n", random.uniform, -0.5, 1.5)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_n, n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=1.0)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", ga_objective)

def main():
    global objective_call_counter
    objective_call_counter = 0
    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    algorithms.eaSimple(pop, toolbox, cxpb=cx_prob, mutpb=mut_prob, ngen=num_generations, 
                        stats=stats, halloffame=hof, verbose=True)

    best_ind = hof[0]
    print(f"Best n found: {best_ind[0]}")
    print(f"Cost function value at best n: {best_ind.fitness.values[0]}")
    df_results = pd.DataFrame({
        'n': n_values,
        'Cost Function': cost_values,
        'Numerical Displacement': numerical_displacements,
        'Numerical Force': numerical_forces,
        'Interpolated Experimental Displacement': interpolated_experimental_displacements,
        'Interpolated Experimental Force': interpolated_experimental_forces
    })
    df_results.to_excel('Optimization_Results_GA.xlsx', index=False)
    print("Optimization results saved to 'Optimization_Results_GA.xlsx'.")

if __name__ == "__main__":
    main()

