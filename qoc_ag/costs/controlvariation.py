"""
controlvariation.py - This module defines a cost function
that penalizes variations of the control parameters.
"""
import autograd.numpy as anp
from numpy import ndarray


class ControlVariation():
    """
    This cost penalizes the variations of the control parameters
    from one `control_eval_step` to the next.

    Parameters
    ----------
    control_num:
        Number of control Hamiltonians
    total_time_steps
    cost_multiplier:
        Weight factor of the cost function; expected < 1
    max_variance:
        This array contains the maximum allowed variance between two time step.
    control_weights:
        Weight factor for each control amplitude
    """
    name = "control_variation"
    requires_step_evaluation = False

    def __init__(self, control_num: int,
                 total_time_steps: int,
                 cost_multiplier: float = 1.,
                 max_variance: ndarray = None,
                 order: int = 1) -> None:
        self.cost_multiplier = cost_multiplier
        self.max_variance = max_variance
        self.diffs_size = control_num * (total_time_steps - order)
        self.order = order
        self.cost_normalization_constant = self.diffs_size * (2 ** self.order)
        self.type = "control_explicitly_related"
        self.control_eval_account = total_time_steps

    def cost(self, controls:ndarray) -> float:
        """
        Compute the penalty.

        Parameters
        ----------
        controls:
            Every control amplitude. Shape is (control_num, toltal_time_steps)
        """
        if self.max_control_norms is None:
            normalized_controls = controls

            # Penalize the square of the absolute value of the difference
            # in value of the control parameters from one step to the next.
            diffs = anp.diff(normalized_controls, axis=0, n=self.order)
            cost = anp.sum(anp.real(diffs * anp.conjugate(diffs)))
            # You can prove that the square of the complex modulus of the difference
            # between two complex values is l.t.e. 2 if the complex modulus
            # of the two complex values is l.t.e. 1 respectively using the
            # triangle inequality. This fact generalizes for higher order differences.
            # Therefore, a factor of 2 should be used to normalize the diffs.
            cost_normalized = cost / self.cost_normalization_constant
        else:
            cost_normalized = 0
            diffs = anp.diff(controls, axis=0, n=self.order)
            for i, max_variance in enumerate(self.max_control_variance):
                diff = diffs[:, i]
                diff_sq = anp.abs(diff)
                penalty_indices = anp.nonzero(diff_sq > max_variance)[0]
                penalized_control = diff_sq[penalty_indices]
                penalty = (penalized_control - max_variance) / penalized_control
                penalty_normalized = penalty / (penalty_indices.shape[0] * len(self.max_control_variance))
                cost_normalized = cost_normalized + anp.sum(penalty_normalized)

        return cost_normalized * self.cost_multiplier
