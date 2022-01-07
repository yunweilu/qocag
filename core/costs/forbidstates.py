"""
forbidstates.py - This module defines a cost function that penalizes
the occupation of a set of forbidden states.
"""

import numpy as np

from core.math.common import conjugate_transpose, conjugate_transpose_ad
import autograd.numpy as anp
from core.math import expmat_der_vec_mul, expmat_vec_mul


class ForbidStates():
    """
    This cost penalizes the occupation of a set of forbidden states.

    Fields:
    cost_multiplier
    cost_normalization_constant
    forbidden_states_count
    forbidden_states_dagger
    name
    requires_step_evalution
    """
    name = "forbid_states"
    requires_step_evaluation = True

    def __init__(self, forbidden_states, total_time_steps,
                 cost_multiplier=1., ):
        """
        See class fields for arguments not listed here.

        Arguments:
        cost_eval_step
        forbidden_states
        system_eval_count
        """
        self.cost_multiplier = cost_multiplier
        self.total_time_steps = total_time_steps
        self.forbidden_states_count = len(forbidden_states)
        self.forbidden_states_dagger = conjugate_transpose_ad(forbidden_states)

        self.type = "control_implicitly_related"
        if len(forbidden_states.shape) is 3:
            self.state_transfer = False
            self.state_count = forbidden_states.shape[1]
            self.forbidden_states = forbidden_states
        else:
            self.dimension = len(self.forbidden_states[0])
            self.state_transfer = True
            self.state_count = 1
            self.forbidden_states = forbidden_states.reshape((self.forbidden_states_count, 1, self.dimension))
        self.cost_normalization_constant = 1 / self.total_time_steps / len(forbidden_states) / (self.state_count ** 2)

    def cost(self, states, mode):
        """
        Compute the penalty.

        Arguments:
        controls
        states
        system_eval_step

        Returns:
        cost
        """
        # The cost is the overlap (fidelity) of the evolved state and each
        # forbidden state.
        if mode is "AG":
            if self.state_transfer is True:
                inner_products = anp.inner(anp.conjugate(self.forbidden_states), states)
                inner_products_square = anp.real(inner_products * anp.conjugate(inner_products))
                cost_value = anp.sum(inner_products_square)
        else:
            if self.state_transfer is True:
                inner_products = anp.inner(anp.conjugate(self.forbidden_states), states)
            else:
                inner_products = anp.trace(anp.matmul(self.forbidden_states_dagger, states))
            inner_products_square = anp.real(inner_products * anp.conjugate(inner_products))
            cost_value = anp.sum(inner_products_square)
            # Normalize the cost for the number of evolving states
            # and the number of times the cost is computed.
        cost_normalized = cost_value * self.cost_normalization_constant
        return cost_normalized

    def gradient_initialize(self, reporter):
        self.final_states = reporter.final_states
        self.back_states = np.zeros_like(self.forbidden_states, dtype="complex_")
        for i in range(len(self.inner_products)):
            for j in range(len(self.inner_products[i])):
                self.back_states[i] = self.forbidden_states[i][j] * self.inner_products[i][j]

    def update_state_back(self, A):
        self.inner_products = np.zeros_like(self.inner_products)
        for i in range(len(self.inner_products)):
            for j in range(len(self.inner_products[i])):
                self.inner_products[i][j] = np.matmul(self.forbidden_states_dagger[j], self.final_states[i])
                self.back_states[i][j] = self.new_state[i][j] + self.inner_products[i][j] * self.forbidden_states[i][j]

    def update_state_forw(self, A, tol):
        self.final_states = expmat_vec_mul(A, tol, self.final_states)

    def gradient(self, A, E, tol):
        grads = 0
        self.new_state = []
        for i in range(len(self.inner_products)):
            self.new_state.append([])
            for j in range(len(self.inner_products[i])):
                b_state, new_state = expmat_der_vec_mul(A, E, tol, self.back_states[i][j])
                self.new_state[i].append(new_state)
                grads = grads + self.cost_multiplier * (2 * np.real(
                    np.matmul(conjugate_transpose(b_state), self.final_states[i]))) / (
                                    self.state_count * self.cost_evaluation_count * self.forbidden_states_count[i])

        return grads
