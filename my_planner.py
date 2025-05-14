# my_planner.py

from planners import Planner
from problems import HealthcareProblem, ResourceType, HealthcareElements # Import necessary enums/classes
from reporter import EventLogReporter, ResourceScheduleReporter # Or your custom reporters
from simulator import EventType # To check lifecycle_state in report

# Define default parameters here, in case best_evolved_params.json is not found
# or for initial testing before running the EA.
# Ensure this structure matches what your EA will produce and save.
DEFAULT_PLANNER_PARAMS = {
    "plan_base_horizon_hours": 48.0,
    "plan_intake_queue_target": 5,
    "plan_replan_improvement_threshold_factor": 0.1,
    # Example for OR scheduling parameters (repeat for A_BED, B_BED, INTAKE, ER_PRACTITIONER)
    "sched_OR_target_utilization": 0.80,
    "sched_OR_queue_increase_thresh": 3,
    "sched_OR_base_level_weekday_day": 4,
    "sched_OR_base_level_weekday_night": 1,
    "sched_OR_base_level_weekend": 1,
    "sched_OR_proactive_increase_demand_factor": 1.2,
    # ... add all other parameters for all resource types
    "sched_A_BED_target_utilization": 0.85,
    # ... etc. for A_BED
    "sched_B_BED_target_utilization": 0.85,
    # ... etc. for B_BED
    "sched_INTAKE_target_utilization": 0.75,
    # ... etc. for INTAKE
    "sched_ER_PRACTITIONER_target_utilization": 0.70,
    # ... etc. for ER_PRACTITIONER
}


class MyEvolvedPlanner(Planner):
    def __init__(self, chromosome_params=None, eventlog_filepath="./temp/my_planner_event_log.csv", data_columns=["diagnosis"]):
        super().__init__()
        
        if chromosome_params is None:
            self.params = DEFAULT_PLANNER_PARAMS
            print("MyEvolvedPlanner: Using DEFAULT parameters.")
        else:
            self.params = chromosome_params
            print(f"MyEvolvedPlanner: Initialized with EA parameters: {self.params}")

        self.eventlog_reporter = EventLogReporter(eventlog_filepath, data_columns)
        self.resource_reporter = ResourceScheduleReporter() # For visualization if needed

        # --- Initialize internal state variables for data collection ---
        # Example: Queue for intake
        self.intake_tasks_waiting = [] # List of case_ids or element objects
        
        # Example: Resource utilization tracking (simplified)
        # You might want more sophisticated tracking (e.g., moving averages over time windows)
        self.resource_busy_time = {res_type: 0 for res_type in ResourceType}
        self.resource_total_scheduled_time = {res_type: 0 for res_type in ResourceType} # If you track this
        self.last_utilization_update_time = 0

        # Example: Track planned admissions (for the NERV KPI if needed, though simulator handles it)
        # self.admission_plans = {} # case_id -> planned_admission_time

        print(f"MyEvolvedPlanner initialized. Logging events to: {eventlog_filepath}")


    def report(self, case_id, element, timestamp, resource, lifecycle_state, data=None):
        # 1. Call standard reporters
        self.eventlog_reporter.callback(case_id, element, timestamp, resource, lifecycle_state, data)
        self.resource_reporter.callback(case_id, element, timestamp, resource, lifecycle_state, data) # If using its graph

        # 2. Implement your custom data collection logic
        # Example: Tracking INTAKE queue
        if element: # Element can be None for some report calls (e.g. CASE_ARRIVAL)
            if element.label == HealthcareElements.INTAKE:
                if lifecycle_state == EventType.ACTIVATE_TASK:
                    if case_id not in self.intake_tasks_waiting: # Avoid duplicates if logic allows
                         self.intake_tasks_waiting.append(case_id)
                elif lifecycle_state == EventType.START_TASK:
                    if case_id in self.intake_tasks_waiting:
                        self.intake_tasks_waiting.remove(case_id)
                elif lifecycle_state == EventType.COMPLETE_TASK: # Or if they leave before starting
                    if case_id in self.intake_tasks_waiting:
                        self.intake_tasks_waiting.remove(case_id)
            
            # Example: Basic resource busy time update
            # A more robust way would be to track start/end of resource usage
            if resource and lifecycle_state == EventType.START_TASK:
                # Logic to start tracking busy time for 'resource.type'
                pass # This might be better handled by looking at changes in busy_resources list
            if resource and lifecycle_state == EventType.COMPLETE_TASK:
                # Logic to accumulate busy time for 'resource.type'
                # self.resource_busy_time[resource.type] += (timestamp - time_task_started_on_resource)
                pass


    def plan(self, cases_to_plan, cases_to_replan, simulation_time):
        planned_cases = []
        
        # --- Use self.params and collected data (e.g., self.intake_tasks_waiting) ---
        
        # Example: Dynamic planning horizon based on intake queue
        current_intake_queue_length = len(self.intake_tasks_waiting)
        target_horizon = self.params.get("plan_base_horizon_hours", 48.0)

        if current_intake_queue_length > self.params.get("plan_intake_queue_target", 5):
            # Increase horizon if queue is long (delay new admissions)
            # This is a simple example; you might want a more nuanced calculation
            target_horizon += (current_intake_queue_length - self.params.get("plan_intake_queue_target", 5)) * 2 # Add 2 hours per excess patient in queue
        
        # Ensure minimum 24-hour lead time
        admission_time_new = simulation_time + max(24.0, target_horizon)
        admission_time_new = round(admission_time_new * 2) / 2 # Round to nearest half hour, for example

        for case_id in cases_to_plan:
            planned_cases.append((case_id, admission_time_new))
            # self.admission_plans[case_id] = admission_time_new # If tracking manually

        # Example: Replanning logic
        # Replanning can be complex due to NERV penalty.
        # Only replan if there's a significant benefit and it doesn't violate constraints.
        replan_improvement_threshold = self.params.get("plan_replan_improvement_threshold_factor", 0.1)
        for case_id in cases_to_replan:
            # Naive: just replan them a bit sooner if possible, but this is likely bad for NERV
            # A better approach would be to see if a new slot is significantly better AND
            # doesn't incur too much NERV. The example NaivePlanner only replanned once.
            # You need to access original_planned_time for NERV calculation - simulator handles NERV.
            # Your goal is to decide if a replan action is beneficial overall.
            
            # For now, a very simple replan: push it to be 24 hours from now if it makes sense
            # This is likely not optimal, just a placeholder.
            potential_replan_time = simulation_time + 24.5 # Slightly more than 24h
            potential_replan_time = round(potential_replan_time * 2) / 2
            
            # Add logic: IF potential_replan_time is "better" (e.g. earlier by a certain amount,
            # or reduces predicted congestion) than original plan, THEN replan.
            # The NERV KPI depends on (original_planned_time - replanning_time).
            # You don't have direct access to original_planned_time here in this simple structure
            # without storing it from previous plan calls or from the problem object (which is forbidden).
            # The simulator tracks this. So your decision is more about "should I initiate a replan action".
            # The example NaivePlanner had a simple `self.replanned_patients` set.
            
            # For now, let's skip complex replanning to keep it focused.
            # planned_cases.append((case_id, potential_replan_time))
            pass

        if planned_cases:
            print(f"Time {simulation_time:.2f}: Planning {len(planned_cases)} cases. Intake Queue: {current_intake_queue_length}. Target Horizon: {target_horizon:.2f}")
        return planned_cases


    def schedule(self, simulation_time):
        scheduled_resources = []
        
        # --- Use self.params and collected data ---
        # Loop through each resource type defined in problems.ResourceType
        for r_type in ResourceType: # ResourceType.OR, ResourceType.A_BED, etc.
            # Get parameters specific to this resource type from self.params
            # Example for OR, you'd need to generalize this or have a loop/dict lookup
            # param_prefix = f"sched_{r_type.value}_" # e.g., "sched_OR_"
            
            # This is an example for ResourceType.OR. You need to make this generic for all types.
            # It's better to structure your self.params to be easily accessible, e.g.,
            # self.params['schedule_policies'][ResourceType.OR]['target_utilization']

            # Simplified example - needs to be robust for all resource types
            # You'll need a mapping from ResourceType enum to the keys in your self.params
            # e.g., if r_type == ResourceType.OR:
            #    base_level_weekday_day = self.params.get("sched_OR_base_level_weekday_day", 3)
            #    ... and so on for other params and other resource types
            
            # --- Determine target number of resources ---
            # This is highly simplified and needs to be fleshed out with:
            # 1. Day of week/time of day logic (weekday_day, weekday_night, weekend)
            # 2. Current utilization / queue length feedback
            # 3. Proactive scheduling based on demand forecast (if you implement one)
            
            # Placeholder: schedule max resources 158 hours ahead (like NaivePlanner for simplicity here)
            # You MUST implement your EA-driven logic here.
            max_resources_map = {
                ResourceType.OR: 5, ResourceType.A_BED: 30, ResourceType.B_BED: 40,
                ResourceType.INTAKE: 4, ResourceType.ER_PRACTITIONER: 9
            }
            target_num_resources = self.params.get(f"sched_{r_type.value}_base_level_weekday_day", max_resources_map[r_type]) # Default to a param or max

            # --- Determine scheduling time ---
            # Must be at least 14 hours ahead. Scheduling further out avoids the "increase-only" rule.
            # The NaivePlanner scheduled 158 hours (next week 8:00) and 168 hours (next week 18:00)
            # Your EA might evolve an optimal lead_time_preference.
            schedule_for_time = simulation_time + 158 # Example: Next week, same day, 8 AM if schedule() is at 18:00

            # Ensure it respects all constraints (max number, increase-only if <158h)
            # The simulator's `problem.check_resource_schedule` will do this,
            # but your logic should ideally try to create valid schedules.
            
            scheduled_resources.append((r_type, schedule_for_time, int(target_num_resources)))

        if scheduled_resources:
            print(f"Time {simulation_time:.2f}: Scheduling resources: {scheduled_resources}")
        return scheduled_resources

    def close_logs(self): # Good practice to close files
        if hasattr(self.eventlog_reporter, 'close') and callable(self.eventlog_reporter.close):
            self.eventlog_reporter.close()
        print("MyEvolvedPlanner logs closed.")