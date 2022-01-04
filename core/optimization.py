from core.optimizers.adam import Adam
from core.models.close_system.parameters import system_parameters
from core.math.initialization import initialize_controls
def grape_schroedinger_discrete(total_time_steps,
                                costs, total_time, H0,H_control,
                                initial_states,
                                impose_control_conditions=None,
                                initial_controls=None,
                                max_iteration_num=1000,
                                log_iteration_step=10,
                                max_control_norms=None,
                                min_error=0,
                                optimizer=Adam(),
                                save_file_path=None,
                                save_intermediate_states=False,
                                save_iteration_step=0,mode='AD'):
    sys_para=system_parameters(total_time_steps,
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
                                save_iteration_step,mode)
    controls=initial_controls()


