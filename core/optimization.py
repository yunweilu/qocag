from core.optimizers.adam import Adam
from core.models.close_system.parameters import system_parameters
from core.math.initialization import initialize_controls
from jax import value_and_grad
import numpy as np

def grape_schroedinger_discrete(total_time_steps,
                                costs, total_time, H0, H_control,
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
                                save_iteration_step=0, mode='AD'):
    """
    This method optimizes the evolution of a set of states under the schroedinger
    equation for time-discrete control parameters.

    Args:
    total_time_steps :: int >= 2 - This value determines where definite values
           of the control parameters are evaluated.
    costs :: iterable(qoc.models.cost.Cost) - This list specifies all
           the cost functions that the optimizer should evaluate. This list
           defines the criteria for an "optimal" control set.
    total_time :: float - This value specifies the duration of the
           system's evolution.
    H0 :: ndarray (hilbert_size x hilbert_size) - System Hamiltonian
    H_control :: ndarray - a set of control Hamiltonian
    initial_states :: ndarray (state_count x hilbert_size x 1)
           - This array specifies the states that should be evolved under the
           specified system. These are the states at the beginning of the evolution.

    impose_control_conditions :: (controls :: (control_eval_count x control_count))
                                    -> (controls :: (control_eval_count x control_count))
           - This function is called after every optimization update. Example uses
           include setting boundary conditions on the control parameters.
    initial_controls :: ndarray (control_num(number of control  channels) x total_time_steps)
           - This array specifies the control parameters at each
           control step. These values will be used to determine the `controls`
           argument passed to the `hamiltonian` function at each time step for
           the first iteration of optimization.
    max_iteration_num:: int - This value determines how many total system
           evolutions the optimizer will perform to determine the
           optimal control set.
    log_iteration_step :: int - This value determines how often qoc logs
           progress to stdout. This value is specified in units of system steps,
           of which there are `control_step_count` * `system_step_multiplier`.
           Set this value to 0 to disable logging.
    max_control_norms :: ndarray (control_count) - This array
           specifies the element-wise maximum norm that each control is
           allowed to achieve. If, in optimization, the value of a control
           exceeds its maximum norm, the control will be rescaled to
           its maximum norm. Note that for non-complex values, this
           feature acts exactly as absolute value clipping.
    min_error :: float - This value is the threshold below which
           optimization will terminate.
    optimizer :: class instance - This optimizer object defines the
           gradient-based procedure for minimizing the total contribution
           of all cost functions with respect to the control parameters.
    save_file_path :: str - This is the full path to the file where
           information about program execution will be stored.
           E.g. "./out/foo.h5"
    save_intermediate_densities :: bool - If this value is set to True,
           qoc will write the densities to the save file after every
           system_eval step.
    save_intermediate_states :: bool - If this value is set to True,
           qoc will write the states to the save file after every
           system_eval step.
    save_iteration_step :: int - This value determines how often qoc
           saves progress to the save file specified by `save_file_path`.
           This value is specified in units of system steps, of which
           there are `control_step_count` * `system_step_multiplier`.
           Set this value to 0 to disable saving.
    mode :: string - The way to get gradients of state or gate related cost.
    "AD" or "AG" which refer to automatic differentiation and analytical gradients.

    Returns:
       result :: qoc.models.schroedingermodels.GrapeSchroedingerResult
       """
    sys_para = system_parameters(total_time_steps,
                                 costs, total_time, H0, H_control,
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
                                 save_iteration_step, mode)
    initial_controls = initialize_controls(total_time_steps, initial_controls, max_control_norms)
    initial_controls=np.ravel(initial_controls)
    # turn to optimizer format which is 1darray
    sys_para.optimizer.run(cost_only())

def cost_only(controls,sys_para):
    control_num = sys_para.control_num
    total_time_steps = sys_para.total_time_steps
    controls = np.reshape(controls, (control_num, total_time_steps))
    # turn the optimizer format to the format given by user
    if sys_para.impose_control_conditions:
        controls = sys_para.impose_control_conditions(controls)
    # impose boundary conditions for control
    error = close_evolution(controls, sys_para)
    # Evaluate the cost function.
    if error <= sys_para.min_error:
        terminate = True
    else:
        terminate = False
    # Determine if optimization should terminate.
    return error, terminate

def cost_gradients(controls, sys_para):
    """
    This function is used to get cost values and gradients by only automatic differentiation
    or combination of analytical gradients and AD
    Args:
        controls :: ndarray - control amplitudes
        sys_para :: class - a class that contains system infomation
    return: cost values and gradients
    """
    control_num=sys_para.control_num
    total_time_steps=sys_para.total_time_steps
    controls=np.reshape(controls,(control_num,total_time_steps))
    # turn the optimizer format to the format given by user
    if sys_para.impose_control_conditions:
        controls = sys_para.impose_control_conditions(controls)
    # impose boundary conditions for control
    if sys_para.mode is "AD":
        cost_value, grads = value_and_grad(close_evolution)(controls, sys_para)
    #   if sys_para.mode is "AG":
    grads = np.ravel(grads)
    # turn to optimizer format which is 1darray
    return cost_value, grads

def close_evolution(controls, sys_para):
    """
    Get cost_values by evolving schrodinger equation.
    Args:
        controls :: ndarray - control amplitudes
        sys_para :: class - a class that contains system infomation
    return: cost values and gradients
    """
    cost_value = 0
    for cost in sys_para.costs:
        if cost.type is "control_explicitly_related":
            cost_value = cost_value + cost.cost(controls)
#    if sys_para.mode is "AG":
#       return cost_value
#    if sys_para.mode is "AD":
