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
                 bandwidths: ndarray,
                 cost_multiplier: float = 1., ) -> None:
        self.cost_multiplier = cost_multiplier
        self.bandwidths = bandwidths
        self.control_num = control_num
        dt = evolution_time / (total_time_steps - 1)
        self.total_time_steps = total_time_steps
        times = anp.linspace(0, dt*total_time_steps, total_time_steps + 1)
        times = anp.delete(times, [len(times) - 1])
        self.freqs = anp.fft.fftfreq(len(times), times[1] - times[0])
        self.type = "control_explicitly_related"
        self.anharmonicity=-0.2

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
        for i, bandwidth in enumerate(self.bandwidths):
            min_bandwidths=bandwidth[0]
            max_bandwidths=bandwidth[1]
            control_fft = anp.fft.fft(controls[i])
            control_fft_sq = anp.abs(control_fft)
            penalty_freq_indices_max = anp.nonzero(anp.abs(self.freqs) >= max_bandwidths)[0]
            penalized_ffts = control_fft_sq[penalty_freq_indices_max]
            penalty = anp.sum(penalized_ffts)
            penalty_freq_indices_min = anp.nonzero(anp.abs(self.freqs) <= min_bandwidths)[0]
            penalized_ffts = control_fft_sq[penalty_freq_indices_min]
            penalty = penalty+ anp.sum(penalized_ffts)
            cost = cost + penalty
        self.cost_value = self.cost_multiplier*cost
        return self.cost_value
