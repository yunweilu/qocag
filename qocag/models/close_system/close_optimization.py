from qocag.optimizers.adam import Adam
from qocag.models.close_system.close_parameters import system_parameters
from qocag.functions.initialization import initialize_controls
from qocag.functions.common import get_H_total
from qocag.functions.save_and_plot import print_heading,print_grads
from autograd import value_and_grad
import numpy as np
from qocag.functions import expmat_vec_mul, expm_pade
import scqubits.settings as settings
from scqubits.utils.cpu_switch import get_map_method
import multiprocessing
import autograd.numpy as anp
from functools import partial
import warnings
warnings.simplefilter("ignore", UserWarning)
settings.MULTIPROC = "pathos"

# Default float type in autograd==float32.
class GrapeSchroedingerResult(object):
    """
    This class encapsulates the result of the
    qoc.core.lindbladdiscrete.grape_schroedinger_discrete
    program.

    Fields:
    best_controls
    best_error
    best_final_states
    best_iteration
    """
    def __init__(self, costs_len=None,save_file_path=None,best_controls=None,
                 best_error=np.finfo(np.float64).max,
                 best_final_states=None,
                 best_iteration=None,):
        """
        See class fields for arguments not listed here.
        """
        super().__init__()
        self.best_controls = best_controls
        self.best_error = best_error
        self.best_final_states = best_final_states
        self.best_iteration = best_iteration
        self.iteration = 0
        self.costs_len=costs_len
        self.best_error_set = np.zeros(costs_len)
        self.local_error_set = np.zeros(costs_len)
        self.cost = []
        for i in range(costs_len):
            self.cost.append([])
        self.control_iter=[]
        self.save_file_path=save_file_path

    def save_data(self,timegrid):
        result={"control_iter":self.control_iter,"cost_iter":np.array(self.cost),"times":timegrid}
        for i in range(self.costs_len):
            self.best_error_set[i]=self.cost[i][self.best_iteration-1]
            self.local_error_set[i] = self.cost[i][self.iteration - 1]
        if self.save_file_path != None:
            np.save(self.save_file_path,result)

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
                                save_intermediate_states=False,save_file_path=None,
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
           - This function==called after every optimization update. Example uses
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
           progress to stdout. This value==specified in units of system steps,
           of which there are `control_step_count` * `system_step_multiplier`.
           Set this value to 0 to disable logging.
    max_control_norms :: ndarray (control_count) - This array
           specifies the element-wise maximum norm that each control is
           allowed to achieve. If, in optimization, the value of a control
           exceeds its maximum norm, the control will be rescaled to
           its maximum norm. Note that for non-complex values, this
           feature acts exactly as absolute value clipping.
    min_error :: float - This value==the threshold below which
           optimization will terminate.
    optimizer :: class instance - This optimizer object defines the
           gradient-based procedure for minimizing the total contribution
           of all cost functions with respect to the control parameters.
    save_file_path :: str - This==the full path to the file where
           information about program execution will be stored.
           E.g. "./out/foo.h5"
    save_intermediate_densities :: bool - If this value==set to True,
           qoc will write the densities to the save file after every
           system_eval step.
    save_intermediate_states :: bool - If this value==set to True,
           qoc will write the states to the save file after every
           system_eval step.
    save_iteration_step :: int - This value determines how often qoc
           saves progress to the save file specified by `save_file_path`.
           This value==specified in units of system steps, of which
           there are `control_step_count` * `system_step_multiplier`.
           Set this value to 0 to disable saving.
    mode :: string - The way to get gradients of state or gate related cost.
    "AD" or "AG" which refer to automatic differentiation and analytical gradients.
    tol :: float - Specify the expected error in propagator expansion
    Returns:
       result :: qoc.models.schroedingermodels.GrapeSchroedingerResult
       """
    noise_operator=None
    noise_spectrum = None
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
                                 save_iteration_step, noise_operator,noise_spectrum,mode, tol)
    initial_controls = initialize_controls(total_time_steps, initial_controls, sys_para.max_control_norms)
    costs_len = len(costs)
    result = GrapeSchroedingerResult(costs_len, save_file_path)
    result.control_iter.append(initial_controls)
    # turn to optimizer format which==1darray
    initial_controls = np.ravel(initial_controls)
    print_heading(result.costs_len)
    sys_para.optimizer.run(cost_only, sys_para.max_iteration_num, initial_controls,
                           cost_gradients, args=(sys_para,result))
    return result

def cost_only(controls, sys_para,result):
    control_num = sys_para.control_num
    total_time_steps = sys_para.total_time_steps
    controls = np.reshape(controls, (control_num, total_time_steps))
    # turn the optimizer format to the format given by user
    if sys_para.impose_control_conditions:
        controls = sys_para.impose_control_conditions(controls)
    # impose boundary conditions for control
    sys_para.only_cost = True
    if sys_para.mode=="AD":
        cost_value = close_evolution(controls, sys_para,result)
    if sys_para.mode=="AG":
        cost_value_ad = close_evolution(controls, sys_para)
        cost_value=cost_value_ad+close_evolution_ag_paral(controls,sys_para)
    # Evaluate the cost function.
    if cost_value <= sys_para.min_error:
        terminate = True
    else:
        terminate = False
    # Determine if optimization should terminate.
    return cost_value, terminate

def cost_gradients(controls, sys_para,result):
    """
    This function==used to get cost values and gradients by only automatic differentiation
    or combination of analytical gradients and AD
    Args:
        controls :: ndarray - control amplitudes
        sys_para :: class - a class that contains system infomation
    return: cost values and gradients
    """

    control_num = sys_para.control_num
    total_time_steps = sys_para.total_time_steps
    total_time = sys_para.total_time
    # turn the optimizer format to the format given by user
    controls = np.reshape(controls, (control_num, total_time_steps))
    times = np.linspace(0, total_time, total_time_steps+1)
    times=np.delete(times, [len(times) - 1])
    result.control_iter.append(controls)
    if sys_para.impose_control_conditions:
        controls = sys_para.impose_control_conditions(controls)
    # impose boundary conditions for control
    if sys_para.mode=="AD":
        cost_value, grads = value_and_grad(close_evolution)(controls, sys_para,result)
    else:
        sys_para.only_cost=False
        cost_value_ad_part, grads_ad_part = value_and_grad(close_evolution)(controls, sys_para,result)
        cost_value_ag_part, grads_ag_part = close_evolution_ag_paral(controls, sys_para,result)
        cost_value = cost_value_ad_part + cost_value_ag_part
        grads = grads_ad_part + grads_ag_part
    grads = np.ravel(grads)
    #print total cost value and norm of grads
    #save control, cost value for each iteration
    result.iteration += 1
    if cost_value <= sys_para.min_error:
        terminate = True
    else:
        terminate = False
    if cost_value < result.best_error:
        result.best_controls = controls
        result.best_error = cost_value
        result.best_iteration=result.iteration
    result.save_data(times)
    print_grads(result.iteration, cost_value, grads,result.local_error_set)
    return grads, terminate

def close_evolution(controls, sys_para,result):
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
    mode = sys_para.mode
    cost_value = 0.
    for i,cost in enumerate(sys_para.costs):
        if cost.type=="control_explicitly_related":
            cost_value = cost_value + cost.cost(controls)
            if type(cost.cost_value)==np.float:
                result.cost[i].append(cost.cost_value)
            else:
                result.cost[i].append(cost.cost_value._value)
    if mode=="AG":
        return cost_value
    if mode=="AD":
        for n in range(total_time_steps):
            time_step = n+1
            H_total = get_H_total(controls, H_controls, H0, time_step)
            propagator=expm_pade(-1j * delta_t * H_total)
            state = anp.transpose(anp.matmul(propagator,anp.transpose(state)))
            for cost in sys_para.costs:
                if cost.type == "control_implicitly_related" and cost.requires_step_evaluation:
                    cost_value = cost_value + cost.cost(state, mode, None,None,None)[0]
        for i,cost in enumerate(sys_para.costs):
            if cost.name == "TargetStateInfidelity":
                infidelity=cost.cost(state, mode, None,None,None)[0]
        for i,cost in enumerate(sys_para.costs):
            if cost.name=="Robustness":
                cost_value = cost_value + cost.cost(controls,sys_para,infidelity)[0]
                if type(cost.cost_value) == np.ndarray:
                    result.cost[i].append(cost.cost_value[0])
                else:
                    result.cost[i].append(cost.cost_value._value[0])
                continue
            if cost.type == "control_implicitly_related" and not cost.requires_step_evaluation:
                cost_value = cost_value + cost.cost(state, mode, None,None,None)[0]
                if type(cost.cost_value) == np.ndarray:
                    result.cost[i].append(cost.cost_value[0])
                else:
                    result.cost[i].append(cost.cost_value._value[0])
        return cost_value

def close_evolution_ag_paral(controls, sys_para,result):
    cost_value=0
    grads_ag=0
    if sys_para.state_transfer==True:
        n = 1
    else:
        n = multiprocessing.cpu_count()
    map_close_evolution_ag = partial(close_evolution_ag, controls, sys_para)
    map_multiprocessing = get_map_method(n)
    sys_para.state_packages = list(map_multiprocessing(map_close_evolution_ag,sys_para.state_packages))
    for cost in sys_para.costs:
        if cost.type != "control_explicitly_related":
            grads_factor = cost.grads_factor(sys_para.state_packages)
            for state_package in sys_para.state_packages:
                state_package[cost.name + "_grads_factor"] = grads_factor
            cost_value = cost_value + cost.cost_collection(sys_para.state_packages[0][cost.name + "_grads_factor"])
    if sys_para.only_cost==True:
        return cost_value
    else:
        map_analytical_grads = partial(analytical_grads, controls, sys_para)
        sys_para.state_packages = list(map_multiprocessing(map_analytical_grads,sys_para.state_packages))
        for cost in sys_para.costs:
            if cost.type != "control_explicitly_related":
                grads_ag = grads_ag + cost.grad_collection(sys_para.state_packages)
        return cost_value,grads_ag


def close_evolution_ag(controls, sys_para, state_package):
    total_time_steps = sys_para.total_time_steps
    H_controls = sys_para.H_controls
    H0 = sys_para.H0
    delta_t = sys_para.total_time / total_time_steps
    tol = sys_para.tol
    mode = sys_para.mode
    state_package['forward_state'] = state_package['initial_state']
    for n in range(total_time_steps):
        time_step = n+1
        H_total = get_H_total(controls, H_controls, H0, time_step)
        state_package['forward_state'] = expmat_vec_mul(-1j * delta_t * H_total, state_package['forward_state'], tol)
        for cost in sys_para.costs:
            if cost.type != "control_explicitly_related" and cost.requires_step_evaluation:
                state_package[cost.name + "_cost_value"] =  cost.cost( state_package['forward_state'],mode,
                                                                                  state_package[cost.name],state_package[cost.name + "_cost_value"]
                                                                       ,n)
    for cost in sys_para.costs:
        if cost.type != "control_explicitly_related" and not cost.requires_step_evaluation:
            state_package[cost.name + "_cost_value"] =  cost.cost( state_package['forward_state'],mode,
                                                                                  state_package[cost.name],state_package[cost.name + "_cost_value"]
                                                                       ,n)

    return state_package


def analytical_grads(controls, sys_para , state_package):
    total_time_steps = sys_para.total_time_steps
    H_controls = sys_para.H_controls
    H0 = sys_para.H0
    delta_t = sys_para.total_time / total_time_steps
    tol = sys_para.tol
    costs=sys_para.costs
    for cost in sys_para.costs:
        if cost.type != "control_explicitly_related":
            state_package[cost.name+"_bs"] = cost.gradient_initialize(state_package[cost.name]
                                                                  ,state_package[cost.name+"_grads_factor"])
    for n in range(total_time_steps):
        time_step = total_time_steps - n
        H_total = get_H_total(controls, H_controls, H0, time_step)
        state_package['forward_state'] = expmat_vec_mul(1j * delta_t * H_total, state_package['forward_state'], tol)
        for cost in costs:
            if cost.type != "control_explicitly_related":
                state_package[cost.name + "_grad_value"] = cost.grads( state_package['forward_state'],state_package[cost.name+"_bs"],1j * delta_t * H_total,
                                                                                  1j * delta_t * H_controls,
                                                                                  state_package[cost.name + "_grad_value"],tol,time_step-1)
                state_package[cost.name + "_bs"] = cost.update_bs(state_package[cost.name],state_package[cost.name+"_grads_factor"],time_step-1)
    return state_package




