"""
controlnorm.py - This module defines a cost function that penalizes
the value of the norm of the control parameters.
"""
import autograd.numpy as anp
from numpy import ndarray
class ControlNorm():
    """
    This cost penalizes control frequencies above a set maximum.

    Parameters
    ----------
    control_num:
        Number of control Hamiltonians
    total_time_steps
    cost_multiplier:
        Weight factor of the cost function; expected < 1
    max_bandwidths:
        This array contains the maximum allowed bandwidth of each control.
    control_weights:
        Weight factor for each control amplitude
    """
    name = "control_norm"
    requires_step_evaluation = False

    def __init__(self, control_num: int,
                 total_time_steps: int,
                 cost_multiplier: float = 1.,
                 max_control_norms: ndarray = None, control_weights: ndarray = None) -> None:

        self.cost_multiplier=cost_multiplier
        self.control_weights = control_weights
        self.controls_size = total_time_steps * control_num
        self.max_control_norms = max_control_norms
        self.total_time_steps=total_time_steps
        self.type="control_explicitly_related"
    
    def cost(self, controls: ndarray) -> float:
        """
        Compute the penalty.

        Parameters
        ----------
        controls:
            Every control amplitude. Shape is (control_num, toltal_time_steps)
        """
        cost_normalized=0
        if self.max_control_norms==None:
        # Weight the controls.
            if self.control_weights is not None:
                controls = controls[:, ] * self.control_weights
        # The cost is the sum of the square of the modulus of the normalized,
        # weighted, controls.
            cost = anp.sum(anp.real(controls * anp.conjugate(controls)))
            cost_normalized = cost / self.controls_size
        else:
            for i, max_norm in enumerate(self.max_control_norms):
                control = controls[:, i]
                control_sq = anp.abs(control)
                penalty_indices = anp.nonzero(control_sq >= max_norm)[0]
                penalized_control = control_sq[penalty_indices]
                penalty = (penalized_control-max_norm)/penalized_control
                if self.control_weights is not None:
                    penalty_normalized=penalty*self.control_weights[penalty_indices]
                else:
                    penalty_normalized = penalty / (penalty_indices.shape[0]* len(self.max_control_norms))
                cost_normalized = cost_normalized + anp.sum(penalty_normalized)
        return cost_normalized * self.cost_multiplier