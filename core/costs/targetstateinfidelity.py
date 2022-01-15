"""
targetstateinfidelity.py - This module defines a cost function that
penalizes the infidelity of an evolved state and a target state.
"""
import numpy as np
from core.math.common import conjugate_transpose_ad
import autograd.numpy as anp
from core.math import expmat_der_vec_mul
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
    name = "TargetStateInfidelity"
    requires_step_evaluation = False
    cost_value=0
    def __init__(self, target_states, cost_multiplier=1.):
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
        self.cost_multiplier=cost_multiplier
        self.cost_normalization_constant=1/(self.state_count**2)
        self.target_states_dagger = conjugate_transpose_ad(target_states)
        self.type = "control_implicitly_related"
        self.cost_format = (1)

    def format(self,control_num,total_time_steps):
        self.total_time_steps=total_time_steps
        self.grad_format=( control_num, self.total_time_steps)
        self.cost_format = (1)

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
            return self.cost_value_ag(forward_state , backward_state )

    def cost_value_ad(self,states):
        if self.state_transfer is True:
            inner_product=anp.inner(anp.conjugate(self.target_states),states)
        else:
            inner_product=anp.trace(anp.matmul(self.target_states_dagger, states))
        inner_product_square = anp.real(inner_product * anp.conjugate(inner_product))
        # Normalize the cost for the number of evolving states
        # and the number of times the cost is computed.
        cost_value = 1- inner_product_square * self.cost_normalization_constant
        return cost_value*self.cost_multiplier

    def cost_value_ag(self, forward_state , backward_state):
        inner_product = np.inner(np.conjugate(backward_state), forward_state)
        cost_value=inner_product
        return cost_value

    def grads_factor(self, state_packages):
        grads_fac = 0.
        for state_package in state_packages:
            grads_fac = grads_fac + state_package[self.name + "_cost_value"]
        return grads_fac

    def cost_collection(self,grads_factor):
        return 1-self.cost_normalization_constant*grads_factor*np.conjugate(grads_factor)*self.cost_multiplier

    def gradient_initialize(self, backward_state, grads_factor):
        return backward_state * grads_factor

    def grads(self, forward_state,backward_state,H_total,H_control,grads,tol,time_step_index,control_index):
        propagator_der_state, updated_bs = expmat_der_vec_mul(H_total, H_control, tol, backward_state)
        self.updated_bs=updated_bs
        grads[control_index][time_step_index] = self.cost_multiplier * (-2 *
            np.inner(np.conjugate(propagator_der_state), forward_state)) / (
                                self.state_count ** 2)
        return grads

    def update_bs(self,target_state,grad_factor,time_step):
        return self.updated_bs

    def grad_collection(self, state_packages):
        grads=np.zeros(self.grad_format)
        for state_package in state_packages:
            grads = grads + state_package[self.name + "_grad_value"]
        return np.real(grads)

#a=1/np.sqrt(2)*np.ones(2)
#cost=TargetStateInfidelity(a)
#print(cost.cost(a,mode='AD'))