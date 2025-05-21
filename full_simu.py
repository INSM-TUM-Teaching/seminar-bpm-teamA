import pickle
from custom_planner import CustomPlanner
from problems import HealthcareProblem
from simulator import Simulator

def select_best_candidate(front):
    """
    Selects the best candidate from the Pareto front.
    You can define "best" by an aggregated metric. For this example,
    we'll simply choose the candidate with the lowest personnel cost (index 3)
    as a simple criterion.
    """
    return min(front, key=lambda c: c.fitness[3])

def run_full_simulation(full_simulation_hours=8760):
    """
    Loads the final Pareto front from the EA run, selects a candidate,
    and runs a full simulation over full_simulation_hours (default is 8760 hours, i.e., one year).
    """
    try:
        with open("final_population.pkl", "rb") as f:
            population, fronts = pickle.load(f)
    except FileNotFoundError:
        print("Error: final_population.pkl file not found. Ensure that you have run your EA main script.")
        return

    # Assume the first front is the Pareto front
    pareto_front = fronts[0]
    
    # Select a candidate to validate (you could choose differently based on your own criteria)
    best_candidate = select_best_candidate(pareto_front)
    print("Selected Candidate Gene for Full Simulation Validation:")
    print(best_candidate.gene)
    print("Short-horizon Fitness:", best_candidate.fitness)

    # Create a planner and simulator for full-scale simulation.
    planner = CustomPlanner(best_candidate.gene)
    problem = HealthcareProblem()
    simulator = Simulator(planner, problem)
    
    print(f"\nRunning full simulation for {full_simulation_hours} hours (1 year)...")
    result = simulator.run(full_simulation_hours)
    
    print("\n--- Full Simulation Results ---")
    for metric, value in result.items():
        print(f"{metric}: {value}")

if __name__ == "__main__":
    run_full_simulation()
