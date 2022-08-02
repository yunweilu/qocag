"""
forbidstates.py - This module defines a cost function that penalizes
the occupation of a set of forbidden states.
"""

import numpy as np
from qocag.functions.common import conjugate_transpose_ad
import autograd.numpy as anp
from qocag.functions import expmat_der_vec_mul

class ForbidStates():
    """
    This cost penalizes the occupation of a set of forbidden states.

    Parameters
    ----------
    cost_multiplier:
        Weight factor of the cost function; expected < 1
    forbidden_states:
        States or gates that are not allowed.
    """
    name = "ForbidStates"
    requires_step_evaluation = True

    def __init__(self, forbidden_states: np.ndarray,
                 cost_multiplier: float = 1., ) -> None:

        self.cost_multiplier = cost_multiplier

        self.forbidden_states_count = len(forbidden_states)
        self.type = "control_implicitly_related"

        if len(forbidden_states.shape)==3:
            self.state_transfer = False
            self.state_count = forbidden_states.shape[1]
            self.forbidden_states = forbidden_states
        else:
            self.dimension = len(forbidden_states[0])
            self.state_transfer = True
            self.state_count = 1
            self.forbidden_states = forbidden_states.reshape((self.forbidden_states_count, 1, self.dimension))

    def format(self, control_num: int, total_time_steps: int) -> None:
        """
        Will get shape of cost values and gradients.
        For this cost function, we store the values for each forbidden state and time step.
        We store gradients for each forbidden state, control and time step.
        The reason==that we evolve each state seperately, so we get each cost value
        and sum over them after evolution. Please check the formula in the paper.

        Parameters
        ----------
        cost_multiplier:
            Weight factor of the cost function; expected < 1
        total_time_steps:
            Number of total time steps
        """
        self.total_time_steps = total_time_steps
        self.cost_normalization_constant = 1 / (
                    self.total_time_steps * len(self.forbidden_states) * (self.state_count ** 2))
        self.cost_format = (self.forbidden_states_count, total_time_steps)
        self.grad_format = (self.forbidden_states_count, control_num, self.total_time_steps)

    def cost(self, forward_state: np.ndarray, mode: str,
             backward_state: np.ndarray, cost_value: np.ndarray, time_step: int) -> np.ndarray:
        """
        Compute the cost. The cost==the overlap of each evolved state and its forbidden state.

        Parameters
        ----------
        forward_state:
            Evolving state in the forward evolution.
        mode:
            The way of getting gradients. "AD" or "AG"
        backward_state:
            Forbidden states
        cost_value:
            Cost values that have shape self.cost_format
        time_step:
            Toltal number of time steps
        """
        if mode=="AD":
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
        if self.state_transfer==True:
            inner_products = anp.inner(anp.conjugate(states), self.forbidden_states)
            inner_products_square = anp.real(inner_products * anp.conjugate(inner_products))
        else:
            inner_products_square = 0
            for forbidden_state in self.forbidden_states:
                inner_products = anp.trace(anp.matmul(conjugate_transpose_ad(states), forbidden_state))
                inner_products_square = inner_products_square + anp.real(inner_products * anp.conjugate(inner_products))

        # Normalize the cost for the number of evolving states
        # and the number of times the cost==computed.
        cost_normalized = inner_products_square * self.cost_normalization_constant
        return cost_normalized * self.cost_multiplier

    def cost_value_ag(self, forward_state: np.ndarray,
                      backward_states: np.ndarray, cost_value: np.ndarray, time_step:int) -> np.ndarray:
        """
        Calculate the cost value.
        Parameters
        ----------
        forward_state:
            Evolving state at the specific time step
        backward_states:
            Forbidden state
        cost_value:
            Overlap between the forward_state and backward_states at each time step
        time_step:
            Specific time step.
        Returns
        -------
            Overlap between the forward_state and backward_states at each time step
        """
        for index, backward_state in enumerate(backward_states):
            inner_product = np.inner(np.conjugate(backward_state), forward_state)
            cost_value[index][time_step] = inner_product
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
        Calculate the cost value for each forbidden states/gate from
         the prefactor of backward propagating state
        Parameters
        ----------
        grads_factors:
            The prefactor of backward propagating state.
        Returns
        -------
        Cost value
        """
        cost_value = 0
        for grads_factor in grads_factors:
            cost_value = cost_value + np.real(np.sum(grads_factor * np.conjugate(grads_factor)))
        return np.real(self.cost_normalization_constant * cost_value * self.cost_multiplier);

    def gradient_initialize(self, backward_state: np.ndarray,
                            grads_factor: np.ndarray) -> np.ndarray:
        """
        Initialize the backward propagating states.
        Parameters
        ----------
        backward_state:
            Forbidden states
        grads_factor
            The prefactor of backward propagating state.
        Returns
        -------
        Backward propagating states Y in the paper
        """
        return backward_state * grads_factor[:, -1:]

    def grads(self, forward_state: np.ndarray, backward_states: np.ndarray,
              H_total: np.ndarray, H_control: np.ndarray, grads:np.ndarray
              , tol: float, time_step_index: int) -> np.ndarray:
        """
        Calculate pieces in gradients expression
        Parameters
        ----------
        forward_state
            Backward propagating states. Start from states at the final time step.
        backward_states
            Backward propagating states. Start from forbidden states.
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
        self.updated_bs = []
        control_number = len(H_control)
        for index, backward_states in enumerate(backward_states):
            states = expmat_der_vec_mul(H_total, H_control, tol, backward_states)
            updated_bs = states[control_number]
            self.updated_bs.append(updated_bs)
            for control_index in range(control_number):
                grads[index][control_index][time_step_index] = self.cost_multiplier * (
                        2 * self.cost_normalization_constant *
                        np.inner(np.conjugate(states[control_index]),
                                 forward_state))
        self.updated_bs = np.array(self.updated_bs)
        return grads

    def update_bs(self, forbidden_state: np.ndarray, grad_factor: np.ndarray,
                  time_step: int) -> np.ndarray:
        """
        Update backward propagating states
        Parameters
        ----------
        forbidden_state:
            Forbidden states
        grad_factor
            Prefactor for updating states
        time_step
            The specif time step
        Returns
        -------
        Updated backward propagating states
        """
        if time_step==0:
            return self.updated_bs
        else:
            return self.updated_bs + grad_factor[:, time_step - 1:time_step ] * forbidden_state

    def grad_collection(self, state_packages:dict) -> np.ndarray:
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
            grads = np.real(grads + state_package[self.name + "_grad_value"])
        return np.sum(grads, axis=0)
