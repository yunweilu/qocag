"""
controlbandwidthmax.py - This module defines a cost function that penalizes all
control frequencies above a specified maximum.
"""
import jax.numpy as jnp
class ControlBandwidthMax():
    """
    This cost penalizes control frequencies above a set maximum.

    Fields:
    max_bandwidths :: ndarray (control_num) - This array contains the maximum allowed bandwidth of each control.
    control_num
    freqs :: ndarray (total_time_steps) - This array contains the frequencies of each of the controls.
    name
    requires_step_evaluation
    
    Example Usage:
    control_num = 1
    total_time = 10 #ns
    total_time_steps = 1000
    MAX_BANDWIDTH_0 = 0.4 # GHz
    MAX_BANDWIDTHS = np.array((MAX_BANDWIDTH_0,))
    COSTS = [ControlBandwidthMax(control_num, total_time_steps,
                                 total_time, MAX_BANDWIDTHS)]
    """
    name = "control_bandwidth_max"
    requires_step_evaluation = False

    def __init__(self, control_num,
                 total_time_steps, evolution_time,
                 max_bandwidths,
                 cost_multiplier=1.,):
        """
        See class fields for arguments not listed here.

        Arguments:
        control_num
        total_time_steps
        evolution_time
        """
        super().__init__(cost_multiplier=cost_multiplier)
        self.max_bandwidths = max_bandwidths
        self.control_num = control_num
        dt = evolution_time / (total_time_steps - 1)
        self.total_time_steps=total_time_steps
        self.freqs = jnp.fft.fftfreq(total_time_steps, d=dt)
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
        cost = 0
        # Iterate over the controls, penalize each control that has
        # frequencies greater than its maximum frequency.
        for i, max_bandwidth in enumerate(self.max_bandwidths):
            control_fft = jnp.fft.fft(controls[:, i])
            control_fft_sq = jnp.abs(control_fft)
            penalty_freq_indices = jnp.nonzero(self.freqs >= max_bandwidth)[0]
            penalized_ffts = control_fft_sq[penalty_freq_indices]
            penalty = jnp.sum(penalized_ffts)
            if penalty<1e-4:
                penalty_normalized=0
            else:
                penalty_normalized = penalty-1e-4 / penalty
            cost = cost + penalty_normalized
        cost_normalized = cost / self.control_num

        return cost_normalized * self.cost_multiplier

