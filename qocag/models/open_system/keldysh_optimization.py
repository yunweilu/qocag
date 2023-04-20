from qocag.models.close_system.close_optimization import GrapeSchroedingerResult
from qocag.optimizers.adam import Adam
from qocag.models.close_system.close_parameters import system_parameters
from qocag.functions.initialization import initialize_controls
from qocag.functions.common import get_H_total
from qocag.functions.common import conjugate_transpose_ad
from qocag.functions.save_and_plot import print_heading,print_grads
from autograd import value_and_grad
import numpy as np
from qocag.functions import expm_pade
import autograd.numpy as anp
from autograd.numpy.fft import fft, fftfreq
import warnings
warnings.filterwarnings("ignore")
class Keldyshresult(GrapeSchroedingerResult):
    def __init__(self, costs_len=None,save_file_path=None,best_controls=None,
                 best_error=np.finfo(np.float64).max,
                 best_final_states=None,
                 best_iteration=None, ):
        super().__init__(costs_len,save_file_path,best_controls,
                 best_error,
                 best_final_states,
                 best_iteration,)
        self.noise_operators=[]

def grape_keldysh_discrete(total_time_steps,
                                costs, total_time, H0, H_controls,
                                initial_states,noise_operator,noise_spectrum,
                                impose_control_conditions=None,
                                initial_controls=None,
                                max_iteration_num=1000,
                                log_iteration_step=10,
                                max_control_norms=None,
                                min_error=0,
                                optimizer=Adam(),
                                save_intermediate_states=False,save_file_path=None,
                                save_iteration_step=0,):
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
                                 save_iteration_step,noise_operator,noise_spectrum,mode="AD",tol=1e-8)
    initial_controls = initialize_controls(total_time_steps, initial_controls, sys_para.max_control_norms)
    costs_len = len(costs)
    result = Keldyshresult(costs_len, save_file_path)
    result.control_iter.append(initial_controls)
    noise_operatorfft=close_evolution(initial_controls, sys_para, result)
    #turn to optimizer format which==1darray
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
    cost_value = close_evolution(controls, sys_para,result)
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
    # turn the optimizer format to the format given by user
    control_num = sys_para.control_num
    total_time_steps=sys_para.total_time_steps
    total_time=sys_para.total_time
    controls = np.reshape(controls, (control_num, total_time_steps))
    times = np.linspace(0, total_time, total_time_steps+1)
    times=np.delete(times, [len(times) - 1])
    result.control_iter.append(controls)
    if sys_para.impose_control_conditions:
        controls = sys_para.impose_control_conditions(controls)
    # impose boundary conditions for control
    if sys_para.mode=="AD":
        cost_value, grads = value_and_grad(cost_calculation)(controls, sys_para,result)
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
def cost_only(controls, sys_para,result):
    """
    Get cost_values by evolving schrodinger equation.
    Args:
        controls :: ndarray - control amplitudes
        sys_para :: class - a class that contains system infomation
    return: cost values and gradients
    """
    total_time_steps=sys_para.total_time_steps
    control_num = sys_para.control_num
    controls = np.reshape(controls, (control_num, total_time_steps))
    # turn the optimizer format to the format given by user
    if sys_para.impose_control_conditions:
        controls = sys_para.impose_control_conditions(controls)
    # impose boundary conditions for control
    cost_value = cost_calculation(controls,sys_para,result)
    if cost_value <= sys_para.min_error:
        terminate = True
    else:
        terminate = False
    # Determine if optimization should terminate.
    return cost_value, terminate

def cost_calculation(controls,sys_para,result):
    noise_operatorfft,U_realized=close_evolution(controls, sys_para,result)
    L_realized=map_second_order(noise_operatorfft,sys_para)
    cost_value=0.
    for i,cost in enumerate(sys_para.costs):
        if cost.type=="control_explicitly_related":
            cost_value = cost_value + cost.cost(controls)
            if type(cost.cost_value)==float:
                result.cost[i].append(cost.cost_value)
            else:
                result.cost[i].append(cost.cost_value._value)
        else:
            cost_value = cost_value + cost.cost(L_realized,U_realized,sys_para.initial_states)
            if type(cost.cost_value)==float:
                result.cost[i].append(cost.cost_value)
            else:
                result.cost[i].append(cost.cost_value._value)
    return cost_value[0]
def close_evolution(controls, sys_para,result):
    total_time_steps = sys_para.total_time_steps
    result.noise_operators=[]
    H_controls = sys_para.H_controls
    H0 = sys_para.H0
    dim=len(H0)
    state = np.identity(dim)
    delta_t = sys_para.total_time / total_time_steps
    for n in range(total_time_steps):
        time_step = n+1
        H_total = get_H_total(controls, H_controls, H0, time_step)
        propagator=expm_pade(-1j * delta_t * H_total)
        state = anp.matmul(propagator,state)
        result.noise_operators.append(anp.matmul(conjugate_transpose_ad(state),anp.matmul(sys_para.noise_operator,state)))
    noise_operatorfft=fft(anp.array(result.noise_operators),axis=0)/total_time_steps
    return noise_operatorfft,state

def map_second_order(noise_operatorfft,sys_para):
    total_time=sys_para.total_time
    noise_spectrum=sys_para.noise_spectrum
    omega_p=2*np.pi/total_time
    H0=sys_para.H0
    dim=len(H0)
    I=np.identity(dim)
    L=super_operator_k(noise_operatorfft[0])*noise_spectrum(0)
    a=len(noise_operatorfft)
    print(noise_operatorfft[10])
    if a%2==0:
        for i in range(1, int(a/ 2)):
            L = L + super_operator_k(noise_operatorfft[i]) * noise_spectrum(i * omega_p)
            L = L + super_operator_k(noise_operatorfft[-i]) * noise_spectrum(-i * omega_p)
        L += super_operator_k(noise_operatorfft[-(int(a / 2))]) * noise_spectrum(-int(a/ 2) * omega_p)
    else:
        for i in range(1, int(len(noise_operatorfft) / 2)):
            L = L + super_operator_k(noise_operatorfft[i]) * noise_spectrum(i * omega_p)
            L = L + super_operator_k(noise_operatorfft[-i]) * noise_spectrum(-i * omega_p)
    L=total_time*L
    return L
def super_operator_k(x):
    dim=len(x)
    I=anp.identity(dim)
    A=anp.matmul(dagger(x),x)
    return anp.kron(x,anp.conjugate(x))-1/2*(anp.kron(A,I)+anp.kron(I,anp.conjugate(A)))
def dagger(A):
    return conjugate_transpose_ad(A)
