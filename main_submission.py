from problems import HealthcareProblem
from simulator import Simulator
from my_planner import MyEvolvedPlanner # Your custom planner
import json
import os # For path joining

if __name__ == '__main__':
    # Path to the parameters file
    # If you place best_evolved_params.json in the same directory as this script:
    params_file_path = "best_evolved_params.json"
    # Or, if it's in a specific location relative to this script:
    # script_dir = os.path.dirname(__file__) # Gets directory of the current script
    # params_file_path = os.path.join(script_dir, "best_evolved_params.json")


    loaded_params = None
    try:
        with open(params_file_path, "r") as f:
            loaded_params = json.load(f)
        print(f"Successfully loaded parameters from {params_file_path}")
    except FileNotFoundError:
        print(f"ERROR: Parameters file '{params_file_path}' not found.")
        print("Ensure 'best_evolved_params.json' is in the correct location or run the EA first.")
        print("Falling back to MyEvolvedPlanner's default parameters (if any) or expecting it to handle None.")
        # MyEvolvedPlanner's __init__ should handle loaded_params being None and use its internal defaults
    except json.JSONDecodeError:
        print(f"ERROR: Could not decode JSON from '{params_file_path}'. File might be corrupted.")
        print("Falling back to MyEvolvedPlanner's default parameters.")


    # --- This is the part the competition organizers will run ---
    # Make sure your_planner can be initialized correctly even if loaded_params is None
    # (e.g., by having default parameters within MyEvolvedPlanner)
    submission_log_file = "./submission_event_log.csv" # As per competition guidelines for output structure
    
    your_planner = MyEvolvedPlanner(chromosome_params=loaded_params, 
                                    eventlog_filepath=submission_log_file, 
                                    data_columns=["diagnosis"])
    
    problem = HealthcareProblem()
    simulator = Simulator(your_planner, problem)
    result = simulator.run(365*24) # Full year simulation for submission
    
    your_planner.close_logs() # Good practice
    # --- End of competition-critical part ---

    print("\n--- Submission Run Results ---")
    print("KPIs from the full-year simulation for submission:")
    for k, v in result.items():
        print(f"  {k}: {v}")

    # You would then zip your_planner.py, this script (e.g., main_submission.py),
    # the best_evolved_params.json (if needed by your planner at runtime),
    # and your PDF report for submission.
    # Ensure that the main runnable file for the organizers is clearly identified.