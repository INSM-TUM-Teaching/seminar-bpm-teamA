import pygad
import json
import numpy as np
import time
import os

# Import problem, simulator, and your planner
from problems import HealthcareProblem, ResourceType
from simulator import Simulator
from my_planner import MyEvolvedPlanner, DEFAULT_PLANNER_PARAMS # Make sure my_planner.py and this default dict exist

# --- Ensure temp directory exists ---
if not os.path.exists("./temp"):
    os.makedirs("./temp")

# --- Helper function to map EA solution (list of numbers) to parameter dictionary ---
def solution_to_params(solution_array):
    params = {}
    idx = 0
    try:
        params["plan_base_horizon_hours"] = solution_array[idx]; idx += 1
        params["plan_intake_queue_target"] = int(round(solution_array[idx])); idx += 1
        params["plan_replan_improvement_threshold_factor"] = solution_array[idx]; idx += 1
        
        for r_type_enum in ResourceType:
            r_type_str = r_type_enum.value
            params[f"sched_{r_type_str}_target_utilization"] = solution_array[idx]; idx += 1
            params[f"sched_{r_type_str}_queue_increase_thresh"] = int(round(solution_array[idx])); idx += 1
            params[f"sched_{r_type_str}_base_level_weekday_day"] = int(round(solution_array[idx])); idx += 1
            params[f"sched_{r_type_str}_base_level_weekday_night"] = int(round(solution_array[idx])); idx += 1
            params[f"sched_{r_type_str}_base_level_weekend"] = int(round(solution_array[idx])); idx += 1
            params[f"sched_{r_type_str}_proactive_increase_demand_factor"] = solution_array[idx]; idx += 1
        
        if idx != len(solution_array):
            print(f"!!! WARNING: Parameter mapping length mismatch. Expected {idx} genes, got {len(solution_array)}. Check gene_space and solution_to_params.")
            return DEFAULT_PLANNER_PARAMS.copy() 
    except IndexError:
        print(f"!!! ERROR: IndexError in solution_to_params. solution_array length: {len(solution_array)}, current index: {idx}. Check gene_space consistency.")
        return DEFAULT_PLANNER_PARAMS.copy()
    return params

# --- Fitness Function ---
fitness_eval_count = 0

# --- CONFIGURATION FOR FITNESS EVALUATION ---
SIMULATION_HOURS_FOR_EA_EVAL = 24 * 1  # 90 days for EA training (Adjust if needed, but be consistent with KPI_RANGES)

# **CRITICAL: ESTIMATE THESE RANGES CAREFULLY FOR THE SIMULATION_HOURS_FOR_EA_EVAL DURATION!**
# 1. Run NaivePlanner for SIMULATION_HOURS_FOR_EA_EVAL. Note its KPIs. These are good "worst" benchmarks.
# 2. Run your MyEvolvedPlanner (with params that gave bad WTH) for SIMULATION_HOURS_FOR_EA_EVAL. Note its WTH.
#    Your 'worst' for WTH should be AT LEAST this high, or higher.
# 'best' values should be optimistic but achievable targets for the EA training duration.
KPI_RANGES_FOR_EA_EVAL = {
    # Example for 90 days - REPLACE WITH YOUR ESTIMATES
    'waiting_time_for_admission': {'best': 20000,  'worst': 200000},   # e.g., NaivePlanner gave 150k for 90 days
    'waiting_time_in_hospital':   {'best': 200000, 'worst': 6000000},  # e.g., Your bad evolved planner gave ~4.8M for 90 days
    'nervousness':                {'best': 0,      'worst': 75000},    # e.g., NaivePlanner gave 50k
    'personnel_cost':             {'best': 100000, 'worst': 400000}    # e.g., NaivePlanner gave 300k
}
# --- END OF FITNESS EVALUATION CONFIGURATION ---

def normalize_kpi_for_ea(kpi_name, value):
    try:
        b = KPI_RANGES_FOR_EA_EVAL[kpi_name]['best']
        w = KPI_RANGES_FOR_EA_EVAL[kpi_name]['worst']
    except KeyError:
        print(f"!!! WARNING: KPI '{kpi_name}' not found in KPI_RANGES_FOR_EA_EVAL. Returning 0.5 for normalization.")
        return 0.5

    if (w - b) == 0: # Avoid division by zero
        # If best and worst are the same, any value at 'b' is "best" (0), others are effectively "out of range" (1).
        return 0.0 if abs(value - b) < 1e-6 else 1.0 
    
    norm_val = (value - b) / (w - b) # Normalizes so that 'best' maps to 0, 'worst' maps to 1
    return max(0.0, min(1.0, norm_val)) # Clip to [0, 1] range

def fitness_function(ga_instance_or_solution, solution_if_not_ga, solution_idx_if_not_ga):
    global fitness_eval_count
    current_eval_idx = fitness_eval_count
    fitness_eval_count += 1

    if isinstance(ga_instance_or_solution, pygad.GA):
        solution_array = solution_if_not_ga
    else:
        solution_array = ga_instance_or_solution

    chromosome_params = solution_to_params(solution_array)
    if not chromosome_params: # If solution_to_params returned None due to error
        print(f"Eval {current_eval_idx:03d}: Error in parameter conversion. Penalizing.")
        return -float('inf')
        
    event_log_filename = f"./temp/ea_run_eval_{current_eval_idx}_log.csv"
    
    planner = MyEvolvedPlanner(chromosome_params, event_log_filename, ["diagnosis"])
    problem = HealthcareProblem() 
    simulator = Simulator(planner, problem) 
    
    sim_start_time_mono = time.monotonic()
    try:
        result_kpis = simulator.run(SIMULATION_HOURS_FOR_EA_EVAL)
    except Exception as e:
        print(f"!!! SIMULATION ERROR for eval {current_eval_idx} with params (approx) { {k: round(v,2) if isinstance(v,float) else v for k,v in chromosome_params.items()} }:\n{e}")
        if hasattr(planner, 'close_logs') and callable(planner.close_logs):
            planner.close_logs()
        return -float('inf') 

    sim_duration = time.monotonic() - sim_start_time_mono
    if hasattr(planner, 'close_logs') and callable(planner.close_logs):
        planner.close_logs()

    norm_wta = normalize_kpi_for_ea('waiting_time_for_admission', result_kpis['waiting_time_for_admission'])
    norm_wth = normalize_kpi_for_ea('waiting_time_in_hospital', result_kpis['waiting_time_in_hospital'])
    norm_nerv = normalize_kpi_for_ea('nervousness', result_kpis['nervousness'])
    norm_cost = normalize_kpi_for_ea('personnel_cost', result_kpis['personnel_cost'])
    
    # Score: lower is better (0 is ideal if all normalized KPIs are 0)
    score = (norm_wta + norm_wth + norm_nerv + 3 * norm_cost) / 6.0

    # Fitness: higher is better for PyGAD
    if score < 1e-9: # Handles score being effectively zero (ideal KPIs)
        fitness = 1.0 / 1e-9 
    else:
        fitness = 1.0 / score
    
    # Reduced verbosity for faster EA runs, focus on key metrics
    print(f"Eval {current_eval_idx:03d}: SimDur {sim_duration:5.1f}s. Score {score:6.4f}. Fit {fitness:8.2e}. WTH_norm {norm_wth:4.2f}. COST_norm {norm_cost:4.2f}. Params ~ H:{chromosome_params.get('plan_base_horizon_hours',0):.0f}|OR_Lvl:{chromosome_params.get('sched_OR_base_level_weekday_day',0)}")
    
    return fitness

if __name__ == '__main__':
    # --- DEFINE GENE SPACE (MUST MATCH solution_to_params) ---
    max_resource_map = { "OR": 5, "A_BED": 30, "B_BED": 40, "INTAKE": 4, "ER_PRACTITIONER": 9}
    gene_space = []
    # Plan params
    gene_space.append({'low': 24.0, 'high': 168.0, 'step': 4.0}) # plan_base_horizon_hours (coarser step for faster exploration initially)
    gene_space.append({'low': 0, 'high': 20, 'step': 1})        # plan_intake_queue_target
    gene_space.append({'low': 0.0, 'high': 0.5, 'step': 0.05}) # plan_replan_improvement_threshold_factor
    # Schedule params
    for r_type_enum in ResourceType:
        max_val = max_resource_map[r_type_enum.value]
        min_val_sched = 1 if r_type_enum in [ResourceType.OR, ResourceType.INTAKE, ResourceType.ER_PRACTITIONER] else 0 # Ensure at least 1 for some key resources
        
        gene_space.append({'low': 0.6, 'high': 0.95, 'step': 0.02}) # target_utilization (encourage higher utilization)
        gene_space.append({'low': 1, 'high': 15, 'step': 1})      # queue_increase_thresh
        gene_space.append({'low': min_val_sched, 'high': max_val, 'step': 1}) # base_level_weekday_day
        gene_space.append({'low': 0, 'high': max_val, 'step': 1}) # base_level_weekday_night
        gene_space.append({'low': 0, 'high': max_val, 'step': 1}) # base_level_weekend
        gene_space.append({'low': 1.0, 'high': 2.0, 'step': 0.1}) # proactive_increase_demand_factor (less aggressive initially)
    num_genes = len(gene_space)

    # --- EA CONFIGURATION (ADJUST THESE VALUES) ---
    NUMBER_OF_GENERATIONS = 1     # Increased generations
    SOLUTIONS_PER_POPULATION = 2  # Increased population
    PARENTS_MATING = 2            
    MUTATION_PERCENT = 15         
    SATURATE_GENS_COUNT = 7        # Stop if no improvement for this many generations
    KEEP_ELITISM_COUNT = 2         
    # --- END OF EA CONFIGURATION ---

    stop_criteria_list = []
    if SATURATE_GENS_COUNT and SATURATE_GENS_COUNT > 0:
        stop_criteria_list.append(f"saturate_{SATURATE_GENS_COUNT}")

    ga_instance = pygad.GA(num_generations=NUMBER_OF_GENERATIONS,
                           num_parents_mating=PARENTS_MATING,
                           fitness_func=fitness_function,
                           sol_per_pop=SOLUTIONS_PER_POPULATION,
                           num_genes=num_genes,
                           gene_space=gene_space,
                           parent_selection_type="sss", 
                           keep_elitism=KEEP_ELITISM_COUNT,      
                           crossover_type="single_point", # Can try "two_points" or "uniform"
                           mutation_type="random", # Can try "adaptive_mutation" if available and configured
                           mutation_percent_genes=MUTATION_PERCENT,
                           stop_criteria=stop_criteria_list if stop_criteria_list else None,
                           on_generation=lambda g: print(f"\n--- Generation {g.generations_completed:02d}/{NUMBER_OF_GENERATIONS} --- Best Fitness So Far: {g.best_solutions_fitness[-1] if g.best_solutions_fitness else 'N/A' :.4e} ---")
                           )

    print(f"Starting EA run: {NUMBER_OF_GENERATIONS} generations, {SOLUTIONS_PER_POPULATION} sol/pop.")
    if stop_criteria_list:
        print(f"Stop criteria: {stop_criteria_list}")
    print(f"Fitness evaluations will use {SIMULATION_HOURS_FOR_EA_EVAL}-hour simulations.")
    print(f"Using KPI Ranges: {KPI_RANGES_FOR_EA_EVAL}")
    
    run_start_time = time.monotonic()
    ga_instance.run()
    run_duration = time.monotonic() - run_start_time
    print(f"\nEA run finished in {run_duration/60:.2f} minutes ({run_duration:.2f} seconds).")

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    best_params = solution_to_params(solution)
    
    print("\n===========================================")
    print("Best solution parameters found:")
    if best_params: 
        for k, v_orig in best_params.items():
            v = round(v_orig, 3) if isinstance(v_orig, float) else v_orig
            print(f"  {k}: {v}")
    else:
        print("  Could not retrieve best parameters due to an earlier error in solution_to_params.")
    
    print(f"Best solution fitness (1/score): {solution_fitness:.4e}")
    if solution_fitness > 1e-9 :
        best_score = 1.0 / solution_fitness
        print(f"Best solution score (lower is better): {best_score:.4f}")
    else:
        print(f"Best solution fitness ({solution_fitness:.4e}) suggests an issue or perfect score not representable by 1/fitness.")
    print("===========================================")

    output_params_file = "best_evolved_params.json"
    if best_params: # Only save if params are valid
        try:
            with open(output_params_file, "w") as f:
                json.dump(best_params, f, indent=4)
            print(f"Best parameters saved to {output_params_file}")
        except Exception as e:
            print(f"!!! ERROR saving {output_params_file}: {e}")
    else:
        print(f"Skipping saving parameters to {output_params_file} as best_params is not available (likely due to error in solution_to_params).")

    try:
        plot_save_path = os.path.join(".", "ea_fitness_plot.png")
        ga_instance.plot_fitness(title="EA Fitness Over Generations", save_dir=plot_save_path)
        print(f"Fitness plot saved to {plot_save_path}")
    except Exception as e:
        print(f"!!! ERROR plotting fitness: {e}")


    if best_params: 
        print("\n--- Verifying best solution on full-year simulation (as in submission script) ---")
        full_year_event_log = "./temp/best_solution_full_year_log.csv"
        final_planner = MyEvolvedPlanner(best_params, full_year_event_log, ["diagnosis"])
        final_problem = HealthcareProblem()
        final_simulator = Simulator(final_planner, final_problem)
        
        verify_sim_start_time = time.monotonic()
        try:
            final_kpis = final_simulator.run(365 * 24) 
        except Exception as e:
            print(f"!!! ERROR during full-year verification simulation: {e}")
            final_kpis = None 
        verify_sim_duration = time.monotonic() - verify_sim_start_time
        
        if hasattr(final_planner, 'close_logs') and callable(final_planner.close_logs):
            final_planner.close_logs()

        if final_kpis:
            print(f"Full-year verification simulation took {verify_sim_duration/60:.2f} minutes.")
            print("Final KPIs for best solution (full year):")
            for k, v_orig in final_kpis.items():
                v = round(v_orig, 2) if isinstance(v_orig, float) else v_orig
                print(f"  {k}: {v}")
        else:
            print("Full-year verification could not be completed due to an error.")
    else:
        print("\nSkipping full-year verification as best parameters were not successfully determined.")