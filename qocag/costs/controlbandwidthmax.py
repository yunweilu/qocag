"""
controlbandwidthmax.py - This module defines a cost function that penalizes all
control frequencies above a specified maximum.
"""
import autograd.numpy as anp
from numpy import ndarray

class ControlBandwidthMax():
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
    """
    name = "control_bandwidth_max"
    requires_step_evaluation = False

    def __init__(self, control_num: int,
                 total_time_steps: int, evolution_time: float,
                 max_bandwidths: ndarray,
                 cost_multiplier: float = 1., ) -> None:
        self.cost_multiplier = cost_multiplier
        self.max_bandwidths = max_bandwidths
        self.control_num = control_num
        dt = evolution_time / (total_time_steps - 1)
        self.total_time_steps = total_time_steps
        self.freqs = anp.fft.fftfreq(total_time_steps, d=dt)
        self.type = "control_explicitly_related"

    def cost(self, controls: ndarray) -> float:
        """
        Compute the penalty.

        Parameters
        ----------
        controls:
            Every control amplitude. Shape==(control_num, toltal_time_steps)

        Returns
        -------
        Cost value
        """
        cost = 0
        # Iterate over the controls, penalize each control that has
        # frequencies greater than its maximum frequency.
        for i, max_bandwidth in enumerate(self.max_bandwidths):
            control_fft = anp.fft.fft(controls[i])
            control_fft_sq = anp.abs(control_fft)
            penalty_freq_indices = anp.nonzero(self.freqs >= max_bandwidth)[0]
            penalized_ffts = control_fft_sq[penalty_freq_indices]
            penalty = anp.sum(penalized_ffts)
            if penalty < 1e-4:
                penalty_normalized = 0
            else:
                penalty_normalized = penalty - 1e-4 / penalty
            cost = cost + penalty_normalized
        self.cost_value = self.cost_multiplier*cost / self.control_num*controls[0][0]/controls[0][0]
        return self.cost_value
