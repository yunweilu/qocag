"""
ko_targetstateinfidelity.py - This module defines a cost function that
penalizes the infidelity of an evolved state and a target state.
"""
import numpy as np
from qocag.functions.common import conjugate_transpose_ad
from qocag.functions.matrix_exponential_vector_multiplication import expm_pade
import autograd.numpy as anp


class KOTargetStateInfidelity():
    """
    This cost penalizes the infidelity .
    Parameters
    ----------
    cost_multiplier:
        Weight factor of the cost function; expected < 1
    torbidden_states:
        Target states
    """
    name = "KOTargetStateInfidelity"
    requires_step_evaluation = False

    def __init__(self, target_states: np.ndarray, cost_multiplier :float = 1., phase=True) -> None:
        if len(target_states.shape)>1:
            self.state_transfer = False
            self.state_count = target_states.shape[0]
            self.target_states = target_states
        else:
            self.state_transfer = True
            self.state_count = 1
            self.target_states = np.kron(target_states,target_states)
        self.cost_multiplier = cost_multiplier
        self.cost_normalization_constant = 1 / (self.state_count ** 2)
        self.type = "control_implicitly_related"
        self.cost_format = (1)
        self.phase=phase

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
        self.grad_format = (control_num, self.total_time_steps)
        self.cost_format = (1)

    def cost(self, L_realized:np.ndarray,U_realized:np.ndarray,
                      initial_states:np.ndarray) -> float:
        """
        Calculate the cost value
        Parameters
        ----------
        states:
            Evolving state at the specific time step
        Returns
        -------
        Cost value. Float
        """
        exp_L = expm_pade(L_realized)
        if self.state_transfer==True:
            # L_realized+=anp.identity(dim )
            initial_rho=anp.kron(initial_states,initial_states)
            states=anp.matmul(exp_L,initial_rho)
            U_realized=anp.kron(conjugate_transpose_ad(U_realized),anp.transpose(U_realized))
            target_states=anp.matmul(U_realized,self.target_states)
            target_states=anp.array([target_states])
            fidelity = anp.real(anp.inner(anp.conjugate(target_states), states))
        else:
            target_U_rotating = anp.matmul(conjugate_transpose_ad(self.target_states), U_realized)
            L_target_dag = anp.kron(target_U_rotating, anp.conjugate(target_U_rotating))
            if self.phase==True:
                fidelity = anp.real(self.cost_normalization_constant*anp.array([anp.trace(anp.matmul(L_target_dag,exp_L))]))
            else:
                fidelity = anp.real(
                    self.cost_normalization_constant * anp.array([anp.trace(anp.abs(anp.matmul(L_target_dag, exp_L)))]))
        infidelity=1-fidelity
        self.cost_value=infidelity
        return infidelity

