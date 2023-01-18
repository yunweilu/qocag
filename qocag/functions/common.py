import autograd.numpy as anp
import numpy as np

def get_H_total(controls: np.ndarray, H_controls: np.ndarray,
                H0: np.ndarray, time_step: np.ndarray,fast_control: list) -> np.ndarray:
    """
    Get the total Hamiltonian in the specific time step
    Parameters
    ----------
    controls:
        All control amplitudes
    H_controls:
        Control Hamiltonian
    H0:
        Static system Hamiltonian
    time_step:
        The specific time step

    Returns
    -------
        The total Hamiltonian in the specific time step
    """
    control_number = len(controls)
    H_total = H0
    if fast_control!=None:
        resolution=fast_control[0]
        osc_control=fast_control[1]
        for i in range(control_number):
            H_total = H_total + osc_control[i][time_step-1]*controls[i][(time_step - 1)//resolution] * H_controls[i]
    else:
        for i in range(control_number):
            H_total = H_total + controls[i][time_step - 1] * H_controls[i]
    return H_total


def conjugate_transpose(matrix):
    """
    Compute the conjugate transpose of a matrix.
    Parameters
    ----------
    matrix
        the matrix to compute the conjugate transpose of
    Returns
    -------
        the conjugate transpose of matrix
    """

    return (matrix.transpose()).conjugate()


def conjugate_transpose_ad(matrix):
    """
    Compute the conjugate transpose of a matrix. Automatic differentiation one
    Parameters
    ----------
    matrix
        the matrix to compute the conjugate transpose of
    Returns
    -------
        the conjugate transpose of matrix
    """

    conjugate_transpose_ = anp.conjugate(anp.swapaxes(matrix, -1, -2))

    return conjugate_transpose_
