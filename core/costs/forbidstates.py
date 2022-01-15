"""
forbidstates.py - This module defines a cost function that penalizes
the occupation of a set of forbidden states.
"""

import numpy as np

from core.math.common import conjugate_transpose_ad
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
    name = "ForbidStates"
    requires_step_evaluation = True

    def __init__(self, forbidden_states,
                 cost_multiplier=1., ):
        """
        See class fields for arguments not listed here.

        Arguments:
        cost_eval_step
        forbidden_states
        system_eval_count
        """
        self.cost_multiplier = cost_multiplier

        self.forbidden_states_count = len(forbidden_states)
        self.type = "control_implicitly_related"

        if len(forbidden_states.shape) is 3:
            self.state_transfer = False
            self.state_count = forbidden_states.shape[1]
            self.forbidden_states = forbidden_states
        else:
            self.dimension = len(forbidden_states[0])
            self.state_transfer = True
            self.state_count = 1
            self.forbidden_states = forbidden_states.reshape((self.forbidden_states_count, 1, self.dimension))

    def format(self,control_num,total_time_steps):
        self.total_time_steps = total_time_steps
        self.cost_normalization_constant = 1 / (self.total_time_steps *len(self.forbidden_states) * (self.state_count ** 2))
        self.cost_format=(self.forbidden_states_count , total_time_steps)
        self.grad_format=(self.forbidden_states_count , control_num, self.total_time_steps)
    def cost(self,   forward_state , mode, backward_state , cost_value,time_step):
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
        if mode is "AD":
            return self.cost_value_ad(forward_state)
        else:
            return self.cost_value_ag(forward_state , backward_state ,cost_value,time_step)

    def cost_value_ad(self,states):
        if self.state_transfer is True:
            inner_products = anp.inner(anp.conjugate(states), self.forbidden_states)
        else:
            inner_products_square = 0
            for forbidden_state in self.forbidden_states:
                inner_products=anp.trace(anp.matmul(conjugate_transpose_ad(states), forbidden_state))
                inner_products_square=inner_products_square+anp.real(inner_products * anp.conjugate(inner_products))

        # Normalize the cost for the number of evolving states
        # and the number of times the cost is computed.
        cost_normalized = inner_products_square * self.cost_normalization_constant
        return cost_normalized*self.cost_multiplier

    def cost_value_ag(self, forward_state , backward_states,cost_value,time_step):
        for index,backward_state in enumerate(backward_states):
            inner_product = np.inner(np.conjugate(backward_state), forward_state)
            cost_value[index][time_step]=inner_product
        return cost_value*self.cost_multiplier

    def grads_factor(self, state_packages):
        grads_fac = 0.
        for state_package in state_packages:
            grads_fac = grads_fac + state_package[self.name + "_cost_value"]
        return grads_fac

    def cost_collection(self,grads_factors):
        cost_value=0
        for grads_factor in grads_factors:
            cost_value = cost_value + np.real(np.sum(grads_factor * np.conjugate(grads_factor)))
        return  self.cost_normalization_constant * cost_value * self.cost_multiplier

    def gradient_initialize(self, backward_state, grads_factor):
        return backward_state * grads_factor[:,-1]


    def grads(self, forward_state, backward_states, H_total, H_control, grads, tol, time_step_index, control_index):
        self.updated_bs=[]
        for index, backward_states in enumerate(backward_states):
            propagator_der_state, updated_bs = expmat_der_vec_mul(H_total, H_control, tol, backward_states)
            self.updated_bs.append(updated_bs)
            grads[index][control_index][time_step_index] = self.cost_multiplier * (2 * self.cost_normalization_constant *
                                                                        np.inner(np.conjugate(propagator_der_state),
                                                                                 forward_state))
        self.updated_bs=np.array(self.updated_bs)
        return grads

    def update_bs(self,target_state,grad_factor,time_step):
        return self.updated_bs+grad_factor[:,time_step-1]*target_state

    def grad_collection(self, state_packages):
        grads=np.zeros(self.grad_format)
        for state_package in state_packages:
            grads = np.real(grads + state_package[self.name + "_grad_value"])
        return np.sum(grads,axis=0)
