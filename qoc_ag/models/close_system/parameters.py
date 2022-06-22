import numpy as np


class system_parameters():
    final_states = None

    def __init__(self, total_time_steps,
                 costs, total_time, H0, H_controls,
                 initial_states,
                 impose_control_conditions,
                 initial_controls,
                 max_iteration_num,
                 log_iteration_step,
                 max_control_norms,
                 min_error,
                 optimizer,
                 save_file_path,
                 save_intermediate_states,
                 save_iteration_step, mode, tol):
        self.H_controls = np.array(H_controls)
        self.control_num = len(H_controls)
        self.total_time_steps = total_time_steps
        self.costs = costs
        self.total_time = total_time
        self.H0 = H0
        self.impose_control_conditions = impose_control_conditions
        self.initial_controls = initial_controls
        self.max_iteration_num = max_iteration_num
        self.log_iteration_step = log_iteration_step
        if max_control_norms == None:
            self.max_control_norms = np.ones(self.control_num)
        else:
            self.max_control_norms = max_control_norms
        self.min_error = min_error
        self.optimizer = optimizer
        self.save_file_path = save_file_path
        self.save_intermediate_states = save_intermediate_states
        self.save_iteration_step = save_iteration_step
        self.mode = mode
        self.tol = tol
        self.only_cost = False
        if mode == "AG":
            if len(initial_states.shape) == 1:
                self.dimension = 1
                self.state_transfer = True
                self.initial_states = np.array([initial_states])
            else:
                self.dimension = len(initial_states[0])
                self.state_transfer = False
                self.initial_states = initial_states
            self.classification()
        if mode == "AD":
            self.initial_states = initial_states

    def classification(self):
        self.state_packages = []
        for state_index, initial_state in enumerate(self.initial_states):
            state_package = {}
            state_package['initial_state'] = initial_state
            for cost in self.costs:
                if cost.type != "control_explicitly_related":
                    if cost.name != "ForbidStates":
                        state_package[cost.name] = cost.target_states[state_index]
                    else:
                        state_package[cost.name] = cost.forbidden_states[:, state_index]
                    cost.format(self.control_num, self.total_time_steps)
                    state_package[cost.name + "_cost_value"] = np.zeros(cost.cost_format, dtype=complex)
                    state_package[cost.name + "_grad_value"] = np.zeros(cost.grad_format, dtype=complex)
            self.state_packages.append(state_package)
        for cost in self.costs:
            if cost.type != "control_explicitly_related":
                if cost.name != "ForbidStates":
                    cost.target_states = None
                    cost.target_states_dagger = None
                else:
                    cost.forbidden_states = None
                    cost.forbidden_states_dagger = None
