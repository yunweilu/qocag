class system_parameters():
    def __init__(self, total_time_step,
                                costs, total_time, H0,H_control,
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
                                save_iteration_step,mode):
        self.H_control=H_control
        self.hilbert_size = initial_states[0].shape[0]
        self.control_num=len(H_control)
        self.total_time_step=total_time_step
        self.costs=costs
        self.total_time=total_time
        self.H0=H0
        self.impose_control_conditions=impose_control_conditions
        self.initial_controls=initial_controls
        self.max_iteration_num=max_iteration_num
        self.log_iteration_step=log_iteration_step
        self.max_control_norms=max_control_norms
        self.min_error=min_error
        self.optimizer=optimizer
        self.save_file_path=save_file_path
        self.save_intermediate_states=save_intermediate_states
        self.save_iteration_step=save_iteration_step
        self.mode=mode