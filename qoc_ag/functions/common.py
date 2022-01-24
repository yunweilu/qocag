import autograd.numpy as anp
def get_H_total(controls,H_controls,H0,time_step):
    H_total=H0
    for control,H_control in zip(controls,H_controls):
        H_total=H_total+control[time_step-1]*H_control
    return H_total


def conjugate_transpose(matrix):
    """
    Compute the conjugate transpose of a matrix.
    Args:
    matrix :: numpy.ndarray - the matrix to compute
        the conjugate transpose of
    operation_policy :: qoc.OperationPolicy - what data type is
        used to perform the operation and with which method
    Returns:
    _conjugate_tranpose :: numpy.ndarray the conjugate transpose
        of matrix
    """
    return (matrix.transpose()).conjugate()

def conjugate_transpose_ad(matrix):
    """
    Compute the conjugate transpose of a matrix. for AD
    Args:
    matrix :: numpy.ndarray - the matrix to compute
        the conjugate transpose of
    Returns:
    _conjugate_tranpose :: numpy.ndarray the conjugate transpose
        of matrix
    """
    conjugate_transpose_ = anp.conjugate(anp.swapaxes(matrix,-1,-2))

    return conjugate_transpose_
