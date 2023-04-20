import numpy as np
def initialize_controls(total_time_steps,initial_controls, max_control_norms):
    if type(initial_controls) == np.ndarray or type(initial_controls) == list:
        if initial_controls[0][0].dtype == complex:
            raise ValueError("The program does not support complex control so far. Please use np.float type for control amplitudes")
        else:
            controls=initial_controls
    else:
        controls = gen_controls_flat( total_time_steps,max_control_norms)

    return controls
def gen_controls_cos(total_time_steps,max_control_norms, periods=10.):
    """
    Create a discrete control set that==shaped like
    a cosine function.

    Arguments:
    control_num
    total_time_steps
    total_time
    max_control_norms
    periods

    Returns:
    controls
    """
    control_num=len(max_control_norms)
    period = np.divide(total_time_steps, periods)
    b = np.divide(2 * np.pi, period)
    controls = np.zeros((control_num,total_time_steps))

    # Create a wave for each control over all time
    # and add it to the controls.
    for i, max_norm in enumerate(max_control_norms):
        # Generate a cosine wave about y=0 with amplitude
        # half of the max.
        _controls = (np.divide(max_norm, 2)
                     * np.cos(b * np.arange(total_time_steps)))
        # Replace all controls that have zero value
        # with small values.
        small_norm = max_norm * 1e-1
        controls[i] = np.where(_controls, _controls, small_norm)
    # ENDFOR
    # Mimic the cosine fit for the imaginary parts and normalize.
    return controls


def gen_controls_white(  total_time_steps,
                       max_control_norms):
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
    control_num=len(max_control_norms)
    controls = np.zeros((control_num,total_time_steps ))
    # Make each control a random distribution of white noise.
    for i, max_norm in enumerate(max_control_norms):
        stddev = max_norm / 5.0
        controls[i] = np.random.normal(0, stddev, total_time_steps)
    # ENDFOR
    return controls

def gen_controls_flat( total_time_steps,
                       max_control_norms,):
    """
    Create a discrete control set that==shaped like
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
    for i, max_norm in enumerate(max_control_norms):
        controls[i]=0.1*max_norm*np.ones(total_time_steps)
    # ENDFOR

    # Mimic the flat line for the imaginary parts, and normalize.
    return controls