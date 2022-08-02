"""
targetstateinfidelitytime.py - This module defins a cost function that
penalizes the infidelity of evolved states and their respective target states
at each cost evaluation step.
"""

import numpy as np
from qocag.functions.common import conjugate_transpose_ad
import autograd.numpy as anp
from qocag.functions import expmat_der_vec_mul


class TargetStateInfidelityTime():
    """
    This cost penalizes the infidelity of evolved states
    and their respective target states at each cost evaluation step.
    The intended result==that a lower infidelity is
    achieved earlier in the system evolution.
    Parameters
    ----------
    cost_multiplier:
        Weight factor of the cost function; expected < 1
    target_states:
        Target states
    """
    name = "TargetStateInfidelityTime"
    requires_step_evaluation = True

    def __init__(self, target_states: np.ndarray, cost_multiplier :float = 1.) -> None:

        if len(target_states.shape)==2:
            self.state_transfer = False
            self.state_count = target_states.shape[0]
            self.target_states = target_states
        else:
            self.state_transfer = True
            self.state_count = 1
            self.target_states = np.array([target_states])
        self.cost_multiplier = cost_multiplier
        self.cost_multiplier = cost_multiplier
        self.target_states_dagger = conjugate_transpose_ad(self.target_states)
        self.type = "control_implicitly_related"

    def format(self, control_num, total_time_steps) -> None:
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
        self.cost_normalization_constant = 1 / ((self.state_count ** 2) * total_time_steps)
        self.cost_format = (total_time_steps)
        self.grad_format = (control_num, self.total_time_steps)

    def cost(self, forward_state: np.ndarray, mode: str,
             backward_state: np.ndarray, cost_value: np.ndarray, time_step: int) -> np.ndarray:
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
        # The cost==the infidelity of each evolved state and its target state.
        if mode == "AD":
            return self.cost_value_ad(forward_state)
        else:
            return self.cost_value_ag(forward_state, backward_state, cost_value, time_step)

    def cost_value_ad(self, states: np.ndarray) -> float:
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
        if self.state_transfer == True:
            inner_product = anp.inner(self.target_states.conjugate(), states)
        else:
            inner_product = anp.trace(anp.matmul(self.target_states_dagger, states))
        inner_product_square = anp.real(inner_product * anp.conjugate(inner_product))
        # Normalize the cost for the number of evolving states
        # and the number of times the cost==computed.
        cost_value = (1 - inner_product_square) * self.cost_normalization_constant
        return cost_value * self.cost_multiplier

    def cost_value_ag(self, forward_state: np.ndarray,
                      backward_states: np.ndarray, cost_value: np.ndarray, time_step: int) -> np.ndarray:
        """
        Calculate the cost value.
        Parameters
        ----------
        forward_state:
            Evolving state at the specific time step
        backward_states:
            Target state
        cost_value:
            Overlap between the forward_state and backward_states at each time step
        time_step:
            Specific time step.
        Returns
        -------
            Overlap between the forward_state and backward_states at each time step
        """
        inner_product = np.inner(np.conjugate(backward_states), forward_state)
        cost_value[time_step] = inner_product
        return cost_value

    def grads_factor(self, state_packages: dict) -> float:
        """
        Calculate the prefactor of backward propagating state.
        Parameters
        ----------
        state_packages
            Dictionary that has all infomation
        Returns
        -------
        Prefactor of backward propagating state.
        """
        grads_fac = 0.
        for state_package in state_packages:
            grads_fac = grads_fac + state_package[self.name + "_cost_value"]
        return grads_fac

    def cost_collection(self, grads_factors: np.ndarray) -> float:
        """
        Calculate the cost value from
         the prefactor of backward propagating state
        Parameters
        ----------
        grads_factors:
            The prefactor of backward propagating state.
        Returns
        -------
        Cost value
        """
        cost_value = np.real(np.sum(grads_factors * np.conjugate(grads_factors)))
        return np.real(1 - self.cost_normalization_constant * cost_value * self.cost_multiplier)

    def gradient_initialize(self, backward_state: np.ndarray,
                            grads_factor: np.ndarray) -> np.ndarray:
        """
        Initialize the backward propagating states.
        Parameters
        ----------
        backward_state:
            Target states
        grads_factor
            The prefactor of backward propagating state.
        Returns
        -------
        Backward propagating states Y in the paper
        """
        return backward_state * grads_factor[-1]

    def grads(self, forward_state: np.ndarray, backward_states: np.ndarray,
              H_total: np.ndarray, H_control: np.ndarray, grads: np.ndarray
              , tol: float, time_step_index: int,) -> np.ndarray:
        """
        Calculate pieces in gradients expression
        Parameters
        ----------
        forward_state
            Backward propagating states. Start from states at the final time step.
        backward_states
            Backward propagating states. Start from target states.
        H_total:
            Total hamiltonian in this time step
        H_control
            Control hamiltonian
        grads
            Gradients
        tol
            Error tolerance
        time_step_index:
            Which time step right now
        control_index
            Which control Hamiltonian
        Returns
        -------
        Gradients pieces
        """
        states = expmat_der_vec_mul(H_total, H_control, tol, backward_states)
        control_number = len(H_control)
        self.updated_bs = states[control_number]
        for control_index in range(control_number):
            grads[control_index][time_step_index] = self.cost_multiplier * (-2 * self.cost_normalization_constant *
                                                                        np.inner(np.conjugate(states[control_index]),
                                                                                 forward_state)) / (
                                                        self.state_count ** 2)
        return grads

    def update_bs(self, target_states: np.ndarray, grad_factor: np.ndarray,
                  time_step: int) -> np.ndarray:
        """
        Update backward propagating states
        Parameters
        ----------
        target_states:
            Target states
        grad_factor
            Prefactor for updating states
        time_step
            The specif time step
        Returns
        -------
        Updated backward propagating states
        """
        return self.updated_bs + grad_factor[time_step - 1] * target_states

    def grad_collection(self, state_packages: dict) -> np.ndarray:
        """
        Calculate grads from pieces
        Parameters
        ----------
        state_packages
            Dictionary that has all infomation
        Returns
        -------
        Gradients
        """
        grads = np.zeros(self.grad_format)
        for state_package in state_packages:
            grads = grads + state_package[self.name + "_grad_value"]
        return np.real(grads)
