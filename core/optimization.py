from core.optimizers.adam import Adam
from core.models.close_system.parameters import system_parameters
from core.math.initialization import initialize_controls
from core.math.common import get_H_total
from core.math.autogradutil import value_and_grad
import numpy as np
from jax.config import config
from core.math import expmat_vec_mul, expmat_vec_mul_ad, expmat_der_vec_mul

config.update("jax_enable_x64", True)


# Default float type in Jax is float32.
def grape_schroedinger_discrete(total_time_steps,
                                costs, total_time, H0, H_controls,
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
                                save_iteration_step=0, mode='AD', tol=1e-8):
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
    tol :: float - Specify the expected error in propagator expansion
    Returns:
       result :: qoc.models.schroedingermodels.GrapeSchroedingerResult
       """
    sys_para = system_parameters(total_time_steps,
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
                                 save_iteration_step, mode, tol)
    initial_controls = initialize_controls(total_time_steps, initial_controls, sys_para.max_control_norms)
    initial_controls = np.ravel(initial_controls)
    # turn to optimizer format which is 1darray
    sys_para.optimizer.run(cost_only, sys_para.max_iteration_num, initial_controls,
                           cost_gradients, args=(sys_para,))


def cost_only(controls, sys_para):
    control_num = sys_para.control_num
    total_time_steps = sys_para.total_time_steps
    controls = np.reshape(controls, (control_num, total_time_steps))
    # turn the optimizer format to the format given by user
    if sys_para.impose_control_conditions:
        controls = sys_para.impose_control_conditions(controls)
    # impose boundary conditions for control
    if sys_para.mode is "AD":
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
    control_num = sys_para.control_num
    total_time_steps = sys_para.total_time_steps
    controls = np.reshape(controls, (control_num, total_time_steps))
    # turn the optimizer format to the format given by user
    if sys_para.impose_control_conditions:
        controls = sys_para.impose_control_conditions(controls)
    # impose boundary conditions for control
    if sys_para.mode is "AD":
        cost_value, grads = value_and_grad(close_evolution)(controls, sys_para)
    else:
        cost_value, grads = value_and_grad(close_evolution)(controls, sys_para)
        cost_value_ag , final_state=close_evolution_ag(controls,sys_para,sys_para.initial_state)
        cost_value=cost_value+cost_value_ag
        grads = grads + analytical_grads(controls,sys_para,final_state)
    #   if sys_para.mode is "AG":
    grads = np.ravel(grads)
    # turn to optimizer format which is 1darray
    if cost_value <= sys_para.min_error:
        terminate = True
    else:
        terminate = False
    return grads, terminate


def close_evolution(controls, sys_para):
    """
    Get cost_values by evolving schrodinger equation.
    Args:
        controls :: ndarray - control amplitudes
        sys_para :: class - a class that contains system infomation
    return: cost values and gradients
    """
    total_time_steps = sys_para.total_time_steps
    H_controls = sys_para.H_controls
    H0 = sys_para.H0
    state = sys_para.initial_states
    delta_t = sys_para.total_time / total_time_steps
    tol = sys_para.tol
    mode = sys_para.mode
    cost_value = 0
    for cost in sys_para.costs:
        if cost.type is "control_explicitly_related":
            cost_value = cost_value + cost.cost(controls)
    if mode is "AG":
        return cost_value
    if mode is "AD":
        for n in range(total_time_steps):
            time_step = n
            H_total = get_H_total(controls, H_controls, H0, time_step)
            state = expmat_vec_mul_ad(1j * delta_t * H_total, state, tol)
            for cost in sys_para.costs:
                if cost.type is not "control_explicitly_related" and cost.requires_step_evaluation:
                    cost_value = cost_value + cost.cost(state, mode)
        for cost in sys_para.costs:
            if cost.type is not "control_explicitly_related" and not cost.requires_step_evaluation:
                cost_value = cost_value + cost.cost(state, mode)
        return cost_value
def close_evolution_ag(controls, sys_para,initial_state):
    total_time_steps = sys_para.total_time_steps
    H_controls = sys_para.H_controls
    H0 = sys_para.H0
    state = initial_state
    delta_t = sys_para.total_time / total_time_steps
    tol = sys_para.tol
    mode = sys_para.mode
    cost_value = 0
    for n in range(total_time_steps):
        time_step = n
        H_total = get_H_total(controls, H_controls, H0, time_step)
        state = expmat_vec_mul(1j * delta_t * H_total, state, tol)
        for cost in sys_para.costs:
            if cost.type is not "control_explicitly_related" and cost.requires_step_evaluation:
                cost_value = cost_value + cost.cost(state, mode)
    for cost in sys_para.costs:
        if cost.type is not "control_explicitly_related" and not cost.requires_step_evaluation:
            cost_value = cost_value + cost.cost(state, mode)
    return cost,state

def analytical_grads(controls,sys_para,final_state):
    total_time_steps = sys_para.total_time_steps
    H_controls = sys_para.H_controls
    H0 = sys_para.H0

    delta_t = sys_para.total_time / total_time_steps
    tol = sys_para.tol
    mode = sys_para.mode
    costs=sys_para.costs
    cost_value = 0
    for cost in costs:
        if cost.type == "non-control":
            cost.gradient_initialize(final_state)