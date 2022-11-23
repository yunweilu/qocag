import autograd.numpy as anp
from qocag.functions.common import get_H_total
from qocag.functions import expm_pade
import numpy as np
class Robustness():
    name = "Robustness"
    requires_step_evaluation = False

    def __init__(self, robust_operator: np.ndarray,delta:float, cost_multiplier :float = 1.) -> None:
        self.type = "control_implicitly_related"
        self.cost_multiplier = cost_multiplier
        self.robust_operator = robust_operator
        self.sys_para = 0
        self.delta = delta

    def format(self, control_num, total_time_steps):
        """
        Will get shape of cost values and gradients.
        For this cost function, we store the values at each time step.
        We store gradients for each target state, control and time step.
        The reason is that we evolve each state seperately, so we get each cost value
        and sum over them after evolution. Please check the formula in the paper.
        Parameters
        ----------
        cost_multiplier:
            Weight factor of the cost function; expected < 1
        total_time_steps:
            Number of total time steps
        """
        self.total_time_steps = total_time_steps
        self.cost_format = (1)

    def close_evolution(self,controls, sys_para):
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
        for n in range(total_time_steps):
            time_step = n + 1
            H_total = get_H_total(controls, H_controls, H0, time_step)
            propagator = expm_pade(-1j * delta_t * H_total)
            state = anp.transpose(anp.matmul(propagator, anp.transpose(state)))
        for i,cost in enumerate(sys_para.costs):
            if cost.name == "TargetStateInfidelity":
                return cost.cost_value_ad(state)


    def cost(self, controls,sys_para,infidelity) -> np.ndarray:
        """
        Compute the cost. The cost==the overlap of each evolved state and its target state.

        Parameters
        ----------
        forward_state:
            Evolving state in the forward evolution.
        mode:
            The way of getting gradients. "AD" or "AG"
        backward_state:
            Target states
        cost_value:
            Cost values that have shape self.cost_format
        time_step:
            Toltal number of time steps
        """
        sys_para.H0+= self.delta*self.robust_operator
        perturbed_infidelity=self.close_evolution(controls, sys_para)
        self.cost_value=anp.abs((infidelity-perturbed_infidelity)/self.delta)*self.cost_multiplier**2
        return self.cost_value


