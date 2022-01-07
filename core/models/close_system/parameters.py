import numpy as np


class system_parameters():
    final_states=None
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
        self.H_controls = H_controls
        self.control_num = len(H_controls)
        self.total_time_steps = total_time_steps
        self.costs = costs
        self.total_time = total_time
        self.H0 = H0
        self.impose_control_conditions = impose_control_conditions
        self.initial_controls = initial_controls
        self.max_iteration_num = max_iteration_num
        self.log_iteration_step = log_iteration_step
        if max_control_norms is None:
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
        if len(initial_states.shape) is 1:
            self.dimension = 1
            self.state_transfer = True
            self.initial_states=np.array([initial_states])
        else:
            self.dimension = 2
            self.state_transfer = False
            self.initial_states = initial_states


