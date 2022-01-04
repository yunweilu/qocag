import numpy as np
def initialize_controls(control_num,
                        total_time_steps, total_time,
                        initial_controls, max_control_norms):
    if initial_controls is not None:
        if initial_controls.dtype is np.complex128 or np.complex64 or np.complex256:
            raise ValueError("The program does not support complex control so far. Please use np.float type for control amplitudes")
    if initial_controls is None:
        controls = gen_controls_flat( control_num, total_time_steps,
                                     total_time, max_control_norms)


def gen_controls_cos(complex_controls, control_num, total_time_steps,
                     total_time, max_control_norms, periods=10.):
    """
    Create a discrete control set that is shaped like
    a cosine function.

    Arguments:
    complex_controls
    control_num
    total_time_steps
    total_time
    max_control_norms

    periods

    Returns:
    controls
    """
    period = np.divide(total_time_steps, periods)
    b = np.divide(2 * np.pi, period)
    controls = np.zeros((total_time_steps, control_num))

    # Create a wave for each control over all time
    # and add it to the controls.
    for i in range(control_num):
        # Generate a cosine wave about y=0 with amplitude
        # half of the max.
        max_norm = max_control_norms[i]
        _controls = (np.divide(max_norm, 2)
                     * np.cos(b * np.arange(total_time_steps)))
        # Replace all controls that have zero value
        # with small values.
        small_norm = max_norm * 1e-1
        _controls = np.where(_controls, _controls, small_norm)
        controls[:, i] = _controls
    # ENDFOR

    # Mimic the cosine fit for the imaginary parts and normalize.
    if complex_controls:
        controls = (controls - 1j * controls) / np.sqrt(2)

    return controls


def gen_controls_white(complex_controls, control_num, total_time_steps,
                       total_time, max_control_norms, periods=10.):
    """
    Create a discrete control set of random white noise.

    Arguments:
    complex_controls
    control_num
    total_time_steps
    total_time
    max_control_norms

    periods

    Returns:
    controls
    """
    controls = np.zeros((total_time_steps, control_num))

    # Make each control a random distribution of white noise.
    for i in range(control_num):
        max_norm = max_control_norms[i]
        stddev = max_norm / 5.0
        control = np.random.normal(0, stddev, total_time_steps)
        controls[:, i] = control
    # ENDFOR
    return controls


def gen_controls_flat( total_time_steps,
                       max_control_norms,):
    """
    Create a discrete control set that is shaped like
    a flat line with small amplitude.

    Arguments:
    total_time_steps
    max_control_norms

    Returns:
    controls
    """
    control_num=len(max_control_norms)
    controls = np.zeros((control_num,total_time_steps))
    # Make each control a flat line for all time.
    for max_norm , i in enumerate(max_control_norms):
        controls[i]=max_norm*np.ones(total_time_steps)
    # ENDFOR

    # Mimic the flat line for the imaginary parts, and normalize.
    return controls
