# ea_main.py
import random
import pickle
from candidate import (
    Candidate, random_candidate, non_dominated_sort, calculate_crowding_distance,
    tournament_selection, sbx_crossover, polynomial_mutation, POPULATION_SIZE, N_GENERATIONS
)
from custom_planner import CustomPlanner
from problems import HealthcareProblem
from simulator import Simulator
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt

# --- Live Plotting Setup ---
plt.ion()  # Turn on interactive mode.

# Create a 2x2 subplot figure for the four KPIs
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
(ax1, ax2), (ax3, ax4) = axs  # ax1: WTA, ax2: WTH, ax3: NERV, ax4: COST

# Set titles for each subplot
ax1.set_title("Best Waiting Time for Admission (WTA)")
ax2.set_title("Best Waiting Time in Hospital (WTH)")
ax3.set_title("Best Nervousness (NERV)")
ax4.set_title("Best Personnel Cost (COST)")

# Initialize arrays to store best values per generation
best_wta = []
best_wth = []
best_nerv = []
best_cost = []

# --- Candidate Evaluation Functions ---
def evaluate_candidate(candidate, simulation_hours=1000):
    """
    Evaluate a candidate by building a custom planner from its gene,
    running a simulation, and then obtaining a fitness tuple (WTA, WTH, NERV, COST).
    """
    planner = CustomPlanner(candidate.gene)
    problem = HealthcareProblem()
    simulator = Simulator(planner, problem)
    result = simulator.run(simulation_hours)
    # Expected keys: 'waiting time for_admission', 'waiting_time_in_hospital', 'nervousness', 'personnel_cost'
    wta = result.get('waiting time for_admission', 0)
    wth = result.get('waiting_time_in_hospital', 0)
    nerv = result.get('nervousness', 0)
    cost = result.get('personnel_cost', 0)
    candidate.fitness = (wta, wth, nerv, cost)
    return candidate

def evaluate_population_parallel(population, simulation_hours=1000):
    """
    Evaluate each candidate in parallel using ProcessPoolExecutor.
    Returns a new list of candidates with updated fitness.
    """
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(evaluate_candidate, cand, simulation_hours) for cand in population]
        new_population = [future.result() for future in futures]
    return new_population

def evolve_population(population):
    """
    Perform evolution:
    - Non-dominated sorting,
    - Crowding distance calculation,
    - Tournament selection,
    - Offspring generation via SBX crossover and polynomial mutation.
    """
    fronts = non_dominated_sort(population)
    for front in fronts:
        calculate_crowding_distance(front)
    selected = tournament_selection(population)
    new_population = []
    while len(new_population) < len(population):
        parent1, parent2 = random.sample(selected, 2)
        child1, child2 = sbx_crossover(parent1, parent2)
        child1 = polynomial_mutation(child1)
        child2 = polynomial_mutation(child2)
        new_population.extend([child1, child2])
    return new_population[:len(population)]

def nsga2(simulation_hours=1000):
    # Initialize the population
    population = [random_candidate() for _ in range(POPULATION_SIZE)]
    print(f"Initial population of {len(population)} candidates generated.")
    population = evaluate_population_parallel(population, simulation_hours)
    print("Initial population evaluated.")
    
    # Evolution loop
    for gen in range(N_GENERATIONS):
        print(f"Generation {gen} evaluation starting...")
        population = evolve_population(population)
        population = evaluate_population_parallel(population, simulation_hours)
        
        # Perform non-dominated sorting and extract KPI values from the entire population
        fronts = non_dominated_sort(population)
        # For each KPI, we collect the minimum value among all candidates in the population.
        wta_vals = [cand.fitness[0] for cand in population]
        wth_vals = [cand.fitness[1] for cand in population]
        nerv_vals = [cand.fitness[2] for cand in population]
        cost_vals = [cand.fitness[3] for cand in population]
        
        best_wta.append(min(wta_vals))
        best_wth.append(min(wth_vals))
        best_nerv.append(min(nerv_vals))
        best_cost.append(min(cost_vals))
        
        best_candidate = min(population, key=lambda c: c.fitness)  # aggregated minimal approach
        print(f"Generation {gen}: Best fitness = {best_candidate.fitness}")
        
        # Update live plots every 5 generations (or every generation if desired)
        if gen % 5 == 0:
            # Clear each subplot before redrawing
            ax1.clear()
            ax2.clear()
            ax3.clear()
            ax4.clear()
            
            # Set labels and titles for each subplot anew
            ax1.set_title("Best Waiting Time for Admission (WTA)")
            ax2.set_title("Best Waiting Time in Hospital (WTH)")
            ax3.set_title("Best Nervousness (NERV)")
            ax4.set_title("Best Personnel Cost (COST)")
            ax1.set_xlabel("Generation")
            ax2.set_xlabel("Generation")
            ax3.set_xlabel("Generation")
            ax4.set_xlabel("Generation")
            
            # Plot the evolution trends
            ax1.plot(range(len(best_wta)), best_wta, marker='o', linestyle='-')
            ax2.plot(range(len(best_wth)), best_wth, marker='o', linestyle='-')
            ax3.plot(range(len(best_nerv)), best_nerv, marker='o', linestyle='-')
            ax4.plot(range(len(best_cost)), best_cost, marker='o', linestyle='-')
            
            plt.tight_layout()
            plt.draw()
            plt.pause(0.1)
            
    return population, fronts, best_wta, best_wth, best_nerv, best_cost

def run_EA(simulation_hours=1000):
    return nsga2(simulation_hours)

# --- Main EA Run ---
if __name__ == "__main__":
    # Adjust simulation_hours as needed for your experiments.
    pop, fronts, evolution_wta, evolution_wth, evolution_nerv, evolution_cost = run_EA(simulation_hours=1000)
    
    # Save the final population and Pareto front for later analysis.
    with open("final_population.pkl", "wb") as f:
        pickle.dump((pop, fronts), f)
        
    # Turn off interactive mode and display the final evolution plot.
    plt.ioff()
    plt.show()
    
    print("\n- - - Final Pareto Front (Non-dominated Solutions) - - -")
    for candidate in fronts[0]:
        print("Candidate gene:", candidate.gene)
        print("Fitness (WTA, WTH, NERV, COST):", candidate.fitness)
        print()
    
    # Optionally, save the evolution trends.
    with open("evolution_trend.pkl", "wb") as f:
        pickle.dump({
            "WTA": evolution_wta,
            "WTH": evolution_wth,
            "NERV": evolution_nerv,
            "COST": evolution_cost
        }, f)
