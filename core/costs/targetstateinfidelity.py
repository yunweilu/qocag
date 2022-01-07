"""
targetstateinfidelity.py - This module defines a cost function that
penalizes the infidelity of an evolved state and a target state.
"""
import scqubits.settings as settings
import numpy as np
from functools import partial
from scqubits.utils.cpu_switch import get_map_method
import multiprocessing
from core.math.common import conjugate_transpose,conjugate_transpose_ad
import autograd.numpy as anp
from core.math import expmat_der_vec_mul,expmat_vec_mul
class TargetStateInfidelity():
    """
    This cost penalizes the infidelity of an evolved state
    and a target state.

    Fields:
    cost_multiplier
    name
    requires_step_evaluation
    state_count
    target_states_dagger
    """
    name = "control_implicitly_related"
    requires_step_evaluation = False

    def __init__(self, target_states, cost_multiplier=1.):
        """
        See class fields for arguments not listed here.

        Arguments:
        target_states
        """
        if len(target_states.shape) is 2:
            self.state_transfer = False
            self.state_count = target_states.shape[0]
        else:
            self.state_transfer = True
            self.state_count = 1
        self.cost_multiplier=cost_multiplier
        self.cost_normalization_constant=1/(self.state_count**2)
        self.state_count = target_states.shape[0]

        self.target_states = target_states
        self.target_states_dagger = conjugate_transpose_ad(target_states)
        self.type = "non-control"
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
        # The cost is the infidelity of each evolved state and its target state.
        if mode is "AG":
            if self.state_transfer is True:
                inner_product = np.inner(np.conjugate(self.target_states),states)
                inner_product_square = np.real(inner_product * np.conjugate(inner_product))
                cost_value = 1 - inner_product_square * self.cost_normalization_constant
        else:
            if self.state_transfer is True:
                inner_product=anp.inner(self.target_states,anp.conjugate(states))
                self.inner_products_sum=inner_product
            else:
                inner_product=anp.trace(anp.matmul(self.target_states_dagger, states))
            inner_product_square = anp.real(inner_product * anp.conjugate(inner_product))
            # Normalize the cost for the number of evolving states
            # and the number of times the cost is computed.
            cost_value = 1- inner_product_square * self.cost_normalization_constant
        return cost_value*self.cost_multiplier

    def gradient_initialize(self, final_state):
        self.final_states = final_state
        self.back_states = self.target_state * self.inner_products_sum

    def update_state_forw(self, A,tol):
        if len(self.final_states) >= 2:
            n = multiprocessing.cpu_count()
            func = partial(expmat_vec_mul(), A, tol)
            settings.MULTIPROC = "pathos"
            map = get_map_method(n)
            states_mul = []
            for i in range(len(self.final_states)):
                states_mul.append(self.final_states[i])
            self.final_states = np.array(map(func, states_mul))
        else:
            self.final_states = expmat_vec_mul(A, tol, self.final_states)
    def update_state_back(self, A):
        self.back_states = self.new_state

    def gradient(self, A,E,tol):
        if len(self.final_states) >= 100:
            n = multiprocessing.cpu_count()
            func = partial(expmat_der_vec_mul(), A, E, tol)
            settings.MULTIPROC = "pathos"
            map = get_map_method(n)
            states_mul = []
            for i in range(len(self.back_states)):
                states_mul.append(self.back_states[i])
            states = map(func, states_mul)
            b_state = np.zeros_like(self.back_states)
            self.new_state = np.zeros_like(self.back_states)
            for i in range(len(states)):
                b_state[i] = states[i][0]
                self.new_state[i] = states[i][1]
            grads = 0
            if self.neglect_relative_phase == False:
                for i in range(self.state_count):
                    a=np.matmul(conjugate_transpose(b_state[i]), self.final_states[i])
                    grads = grads + self.cost_multiplier * (-2 * np.real(
                        a)) / (
                                    self.state_count ** 2)
            else:
                for i in range(self.state_count):
                    grads = grads + self.cost_multiplier * (-2 * np.real(
                        np.matmul(conjugate_transpose(b_state[i]), self.final_states[i]))) / (
                                self.state_count)
        else:
            grads = 0
            self.new_state = []
            if not self.neglect_relative_phase:
                for i in range(self.state_count):
                    b_state, new_state = expmat_der_vec_mul(A, E, tol, self.back_states[i])
                    self.new_state.append(new_state)
                    grads = grads + self.cost_multiplier * (-2 * np.real(
                        np.matmul(conjugate_transpose(b_state), self.final_states[i]))) / (
                                    self.state_count ** 2)
            else:
                for i in range(self.state_count):
                    b_state, new_state = expmat_der_vec_mul(A, E, tol, self.back_states[i])
                    self.new_state.append(new_state)
                    grads = grads + self.cost_multiplier * (-2 * np.real(
                        np.matmul(conjugate_transpose(b_state), self.final_states[i]))) / (
                                self.state_count)
        return grads
