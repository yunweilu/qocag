"""
targetstateinfidelitytime.py - This module defins a cost function that
penalizes the infidelity of evolved states and their respective target states
at each cost evaluation step.
"""


import numpy as np

from functools import partial
from scqubits.utils.cpu_switch import get_map_method
import multiprocessing
import scqubits.settings as settings
from core.math.common import conjugate_transpose,conjugate_transpose_ad
import autograd.numpy as anp
from core.math import expmat_der_vec_mul,expmat_vec_mul
from scipy.sparse import bmat
class TargetStateInfidelityTime():
    """
    This cost penalizes the infidelity of evolved states
    and their respective target states at each cost evaluation step.
    The intended result is that a lower infidelity is
    achieved earlier in the system evolution.

    Fields:
    cost_eval_count
    cost_multiplier
    name
    requires_step_evaluation
    state_count
    target_states_dagger
    """
    name = "target_state_infidelity_time"
    requires_step_evaluation = True


    def __init__(self,  target_states,
                 total_time_steps, cost_multiplier=1.,):
        """
        See class fields for arguments not listed here.

        Arguments:
        target_states
        """
        if len(target_states.shape) is 2:
            self.state_transfer = False
            self.state_count = target_states.shape[0]
            self.target_states = target_states
        else:
            self.state_transfer = True
            self.state_count = 1
            self.target_states = np.array([target_states])
        self.cost_multiplier = cost_multiplier
        self.cost_normalization_constant = total_time_steps / (self.state_count ** 2)
        self.cost_multiplier=cost_multiplier
        self.target_states_dagger = conjugate_transpose_ad(target_states)
        self.target_states = target_states
        self.type="control_explicitly_related"

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
                inner_product = np.inner(np.conjugate(self.target_states), states)
                inner_product_square = np.real(inner_product * np.conjugate(inner_product))
                cost_value = 1 - inner_product_square * self.cost_normalization_constant
        else:
            if self.state_transfer is True:
                inner_product=anp.inner(self.target_states.conjugate(),states)
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

    def update_state_back(self, A):
        self.back_states = self.new_state
        if self.neglect_relative_phase == False:
            for i in range(self.state_count):
                self.back_states[i] = self.back_states[i]+self.inner_products_sum[self.i]*self.target_states[i]
            self.i=self.i-1
        else:
            self.inner_products = np.matmul(self.target_states_dagger, self.final_states)[:, 0, 0]
            for i in range(self.state_count):
                self.back_states[i] = self.back_states[i] + self.inner_products[i] * self.target_states[i]
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

    def gradient(self, A, E,tol):
        if len(self.final_states) >= 2:
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
            grads = 0
            for i in range(len(states)):
                b_state[i] = states[i][0]
                self.new_state[i] = states[i][1]
            if self.neglect_relative_phase == False:
                for i in range(self.state_count):
                    grads = grads + self.cost_multiplier * (-2 * np.real(
                        np.matmul(conjugate_transpose(b_state[i]), self.final_states[i]))) / (
                                        (self.state_count ** 2) * self.cost_eval_count)
            else:
                for i in range(self.state_count):
                    grads = grads + self.cost_multiplier * (-2 * np.real(
                        np.matmul(conjugate_transpose(b_state[i]), self.final_states[i]))) / (
                                    self.state_count * self.cost_eval_count)
        else:
            grads = 0
            self.new_state = []
            if self.neglect_relative_phase == False:
                for i in range(self.state_count):
                    b_state, new_state = expmat_der_vec_mul(A, E, tol, self.back_states[i])
                    self.new_state.append(new_state)
                    a = expmat_der_vec_mul(b_state)
                    b = self.final_states[i]
                    c = np.matmul(a, b)
                    grads = grads + self.cost_multiplier * (-2 * np.real(
                        np.matmul(conjugate_transpose(b_state), self.final_states[i]))) / (
                                        (self.state_count ** 2) * self.cost_eval_count)
            else:
                for i in range(self.state_count):
                    b_state, new_state = expmat_der_vec_mul(A, E, tol, self.back_states[i])
                    self.new_state.append(new_state)
                    grads = grads + self.cost_multiplier * (-2 * np.real(
                        np.matmul(conjugate_transpose(b_state), self.final_states[i]))) / (
                                    self.state_count * self.cost_eval_count)
        return grads