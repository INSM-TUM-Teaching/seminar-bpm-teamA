# custom_planner.py
from planners import Planner
from problems import ResourceType
from reporter import EventLogReporter, ResourceScheduleReporter

class CustomPlanner(Planner):
    def __init__(self, candidate_params, eventlog_file="./temp/event_log.csv", data_columns=["diagnosis"]):
        super().__init__()
        # Planning offsets from candidate parameters.
        self.p1 = candidate_params[0]  # New-case planning offset (hours)
        self.p2 = candidate_params[1]  # Replan offset (hours)
        
        # Decode scheduling fractions for each resource type.
        self.or_im = max(round(5 * candidate_params[2]), 5)   # Immediate OR (baseline 5)
        self.or_lt = round(5 * candidate_params[3])            # Long-term OR
        self.ab_im = max(round(30 * candidate_params[4]), 30)   # Immediate A_BED (baseline 30)
        self.ab_lt = round(30 * candidate_params[5])           # Long-term A_BED
        self.bb_im = max(round(40 * candidate_params[6]), 40)   # Immediate B_BED (baseline 40)
        self.bb_lt = round(40 * candidate_params[7])           # Long-term B_BED
        self.in_im = max(round(4 * candidate_params[8]), 4)      # Immediate INTAKE (baseline 4)
        self.in_lt = round(4 * candidate_params[9])            # Long-term INTAKE
        self.er_im = max(round(9 * candidate_params[10]), 9)    # Immediate ER_PRACTITIONER (baseline 9)
        self.er_lt = round(9 * candidate_params[11])           # Long-term ER_PRACTITIONER
        
        # Initialize reporters.
        self.eventlog_reporter = EventLogReporter(eventlog_file, data_columns)
        self.resource_reporter = ResourceScheduleReporter()
        self.replanned_patients = set()

    def report(self, case_id, element, timestamp, resource, lifecycle_state, data=None):
        self.eventlog_reporter.callback(case_id, element, timestamp, resource, lifecycle_state)
        self.resource_reporter.callback(case_id, element, timestamp, resource, lifecycle_state, data)

    def plan(self, cases_to_plan, cases_to_replan, simulation_time):
        planned_cases = []
        next_plannable_time = round(simulation_time + self.p1)
        next_replannable_time = round(simulation_time + self.p2)
        for case_id in cases_to_plan:
            planned_cases.append((case_id, next_plannable_time))
        for case_id in cases_to_replan:
            if case_id not in self.replanned_patients:
                planned_cases.append((case_id, next_replannable_time))
                self.replanned_patients.add(case_id)
        return planned_cases

    def schedule(self, simulation_time):
        scheduled_resources = []
        immediate_time = simulation_time + 14
        long_term_time = simulation_time + 158
        scheduled_resources.append((ResourceType.OR, immediate_time, self.or_im))
        scheduled_resources.append((ResourceType.OR, long_term_time, self.or_lt))
        scheduled_resources.append((ResourceType.A_BED, immediate_time, self.ab_im))
        scheduled_resources.append((ResourceType.A_BED, long_term_time, self.ab_lt))
        scheduled_resources.append((ResourceType.B_BED, immediate_time, self.bb_im))
        scheduled_resources.append((ResourceType.B_BED, long_term_time, self.bb_lt))
        scheduled_resources.append((ResourceType.INTAKE, immediate_time, self.in_im))
        scheduled_resources.append((ResourceType.INTAKE, long_term_time, self.in_lt))
        scheduled_resources.append((ResourceType.ER_PRACTITIONER, immediate_time, self.er_im))
        scheduled_resources.append((ResourceType.ER_PRACTITIONER, long_term_time, self.er_lt))
        return scheduled_resources
