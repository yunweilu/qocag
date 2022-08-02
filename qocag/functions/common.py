import autograd.numpy as anp
import numpy as np

def get_H_total(controls: np.ndarray, H_controls: np.ndarray,
                H0: np.ndarray, time_step: np.ndarray) -> np.ndarray:
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
    H_total = H0
    for control, H_control in zip(controls, H_controls):
        H_total = H_total + control[time_step - 1] * H_control
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
