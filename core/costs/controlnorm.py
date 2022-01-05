"""
controlnorm.py - This module defines a cost function that penalizes
the value of the norm of the control parameters.
"""
import jax.numpy as jnp
class ControlNorm():
    """
    This cost penalizes the value of the norm of the control parameters.

    Fields:
    control_weights :: ndarray (total_time_steps x control_num)
        - These weights, each of which should be no greater than 1,
        represent the factor by which each control's magnitude is penalized.
        If no weights are specified, each control's magnitude is penalized
        equally.
    controls_size
    cost_multiplier
    max_control_norms
    name
    requires_step_evaluation
    """
    name = "control_norm"
    requires_step_evaluation = False

    def __init__(self, control_num,
                 total_time_steps,
                 control_weights=None,
                 cost_multiplier=1.,
                 max_control_norms=None,):
        """
        See class fields for arguments not listed here.

        Arguments:
        control_num
        total_time_steps
        """
        super().__init__(cost_multiplier=cost_multiplier)
        self.control_weights = control_weights
        self.controls_size = total_time_steps * control_num
        self.max_control_norms = max_control_norms
        self.total_time_steps=total_time_steps
        self.type="control_explicitly_related"
    
    def cost(self, controls):
        """
        Compute the penalty.

        Arguments:
        controls
        states
        system_eval_step

        Returns:
        cost
        """
        # Normalize the controls.
        cost_normalized=0
        if self.max_control_norms==None:
        # Weight the controls.
            if self.control_weights is not None:
                controls = controls[:, ] * self.control_weights
        # The cost is the sum of the square of the modulus of the normalized,
        # weighted, controls.
            cost = jnp.sum(jnp.real(controls * jnp.conjugate(controls)))
            cost_normalized = cost / self.controls_size
        else:
            for i, max_norm in enumerate(self.max_control_norms):
                control = controls[:, i]
                control_sq = jnp.abs(control)
                penalty_indices = jnp.nonzero(control_sq >= max_norm)[0]
                penalized_control = control_sq[penalty_indices]
                penalty = (penalized_control-max_norm)/penalized_control
                if self.control_weights is not None:
                    penalty_normalized=penalty*self.control_weights[penalty_indices]
                else:
                    penalty_normalized = penalty / (penalty_indices.shape[0]* len(self.max_control_norms))
                cost_normalized = cost_normalized + jnp.sum(penalty_normalized)
        return cost_normalized * self.cost_multiplier