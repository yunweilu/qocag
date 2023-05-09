"""
Action of matrix exponential and action of propagator derivative on a vector or a set of basis
"""
import scipy.sparse.linalg
from scipy.sparse import bmat, isspmatrix
import numpy as np
def expmat_der_vec_mul(A, E, tol, state):
    """
        Calculate the action of propagator derivative.
        First we construct auxiliary matrix and vector. Then use expm_multiply function.
        Arg:
        A :: numpy.ndarray - Total Hamiltonian
        E :: numpy.ndarray - Control Hamiltonian
        state :: numpy.ndarray
        Returns:
        numpy.ndarray,numpy.ndarray - vector for gradient calculation, updated state
        """
    state=np.complex128(state)
    if tol==None:
        tol=2**-53
    control_number = len(E)
    HILBERT_SIZE = state.shape[0]
    final_matrix = []
    for i in range(control_number+1):
        final_matrix.append([])
    if isspmatrix(A) == False:
        for i in range(control_number + 1):
            raw_matrix = []
            if i == 0:
                raw_matrix = raw_matrix + [A]
            else:
                raw_matrix = raw_matrix + [np.zeros_like(A)]
            for j in range(1, control_number + 1):
                if j == i and i != 0:
                    raw_matrix = raw_matrix + [A]
                elif j == control_number and j != i:
                    raw_matrix = raw_matrix + [E[i]]
                    x=j-1
                else:
                    raw_matrix = raw_matrix + [np.zeros_like(A)]
            final_matrix[i] = raw_matrix
        final_matrix = np.block(final_matrix)
    else:
        for i in range(control_number+1):
            raw_matrix = []
            if i == 0:
                raw_matrix=raw_matrix+[A]
            else:
                raw_matrix = raw_matrix+[None]
            for j in range(1,control_number+1):
                if j == i and i != 0:
                    raw_matrix = raw_matrix+[A]
                elif j == control_number  and j!=i:
                    raw_matrix = raw_matrix+[E[i]]
                else:
                    raw_matrix = raw_matrix+[None]
            final_matrix[i] = raw_matrix
        final_matrix=scipy.sparse.bmat(final_matrix)
    state = state.flatten()
    state0 = np.zeros_like(state)
    for i in range(control_number):
        state = np.block([state0, state])
    state = expmat_vec_mul(final_matrix, state,tol)
    states = []
    for i in range(control_number+1):
        states.append(state[HILBERT_SIZE*i:HILBERT_SIZE*(i+1)])
    return np.array(states)


"""Compute the action of the matrix exponential."""


def _exact_inf_norm(A):
    if scipy.sparse.isspmatrix(A):
        return max(abs(A).sum(axis=1).flat)
    else:
        return np.linalg.norm(A, np.inf)

def trace(A):
    """
    Args:
    A :: numpy.ndarray - matrix
    Returns:
    trace of A
    """
    if scipy.sparse.isspmatrix(A):
        return A.diagonal().sum()
    else:
        return np.trace(A)

def ident_like(A):
    """
    Args:
    A :: numpy.ndarray - matrix
    Returns:
    Identity matrix which has same dimension as A
    """
    if scipy.sparse.isspmatrix(A):
        return scipy.sparse.construct.eye(A.shape[0], A.shape[1],
                                          dtype=A.dtype, format=A.format)
    else:
        return np.eye(A.shape[0], A.shape[1], dtype=A.dtype)

def get_s(A, tol):
    s = 1
    a = _exact_inf_norm(A)
    while (1):
        norm_A = a / s
        max_term_notation = np.floor(norm_A)
        max_term = 1
        for i in range(1, np.int64(max_term_notation)):
            max_term = max_term * norm_A / i
            if max_term >= 10 ** 16:
                break
        if 10 ** -16 * max_term <= tol:
            break
        s = s + 1
    return s

def get_s(A, tol):
    s = 1
    a = _exact_inf_norm(A)
    while (1):
        norm_A = a / s
        max_term_notation = np.floor(norm_A)
        max_term = 1
        for i in range(1, np.int64(max_term_notation)):
            max_term = max_term * norm_A / i
            if max_term >= 10 ** 16:
                break
        if 10 ** -16 * max_term <= tol:
            break
        s = s + 1
    return s
def expmat_vec_mul(A, B, tol=None):
    """
    Compute the exponential matrix and vector multiplication e^(A) B.
    Args:
    A :: numpy.ndarray - matrix
    b :: numpy.ndarray - vector or a set of basis vectors
    tol :: numpy.float64 - expected error
    Returns:
    f :: numpy.ndarray - Approximation of e^(A) b
    """
    ident = ident_like(A)
    n = A.shape[0]
    mu = trace(A) / float(n)
    # Why mu? http://eprints.ma.man.ac.uk/1591/, section 3.1
    A = A - mu * ident
    if tol == None:
        tol = 1e-16
    s = get_s(A, tol)
    F = B
    c1 = _exact_inf_norm(B)
    j = 0
    while (1):
        coeff = s * (j + 1)
        B = A.dot(B) / coeff
        c2 = _exact_inf_norm(B)
        F = F + B
        if (c1 + c2) < tol:
            m = j + 1
            break
        c1 = c2
        j = j + 1
    B = F
    for i in range(1, int(s)):
        for j in range(m):
            coeff = s * (j + 1)
            B = A.dot(B) / coeff
            F = F + B
        B = F
    return F

import autograd.numpy as anp
import numpy as np


### EXPM IMPLEMENTATION DUE TO HIGHAM 2005 ###

# Pade approximants from algorithm 2.3.
B = (
    64764752532480000,
    32382376266240000,
    7771770303897600,
    1187353796428800,
    129060195264000,
    10559470521600,
    670442572800,
    33522128640,
    1323241920,
    40840800,
    960960,
    16380,
    182,
    1,
)

def one_norm(a):
    """
    Return the one-norm of the matrix.

    References:
    [0] https://www.mathworks.com/help/dsp/ref/matrix1norm.html

    Arguments:
    a :: ndarray(N x N) - The matrix to compute the one norm of.

    Returns:
    one_norm_a :: float - The one norm of a.
    """
    return anp.max(anp.sum(anp.abs(a), axis=0))

def pade3(a, i):
    a2 = anp.matmul(a, a)
    u = anp.matmul(a, B[2] * a2) + B[1] * a
    v = B[2] * a2 + B[0] * i
    return u, v

def pade5(a, i):
    a2 = anp.matmul(a, a)
    a4 = anp.matmul(a2, a2)
    u = anp.matmul(a, B[5] * a4 + B[3] * a2) + B[1] * a
    v = B[4] * a4 + B[2] * a2 + B[0] * i
    return u, v

def pade7(a, i):
    a2 = anp.matmul(a, a)
    a4 = anp.matmul(a2, a2)
    a6 = anp.matmul(a2, a4)
    u = anp.matmul(a, B[7] * a6 + B[5] * a4 + B[3] * a2) + B[1] * a
    v = B[6] * a6 + B[4] * a4 + B[2] * a2 + B[0] * i
    return u, v

def pade9(a, i):
    a2 = anp.matmul(a, a)
    a4 = anp.matmul(a2, a2)
    a6 = anp.matmul(a2, a4)
    a8 = anp.matmul(a2, a6)
    u = anp.matmul(a, B[9] * a8 + B[7] * a6 + B[5] * a4 + B[3] * a2) + B[1] * a
    v = B[8] * a8 + B[6] * a6 + B[4] * a4 + B[2] * a2 + B[0] * i
    return u, v

def pade13(a, i):
    a2 = anp.matmul(a, a)
    a4 = anp.matmul(a2, a2)
    a6 = anp.matmul(a2, a4)
    u = anp.matmul(a, anp.matmul(a6, B[13] * a6 + B[11] * a4 + B[9] * a2) + B[7] * a6 + B[5] * a4 + B[3] * a2) + B[
        1] * a
    v = anp.matmul(a6, B[12] * a6 + B[10] * a4 + B[8] * a2) + B[6] * a6 + B[4] * a4 + B[2] * a2 + B[0] * i
    return u, v


# Valid pade orders for algorithm 2.3.
PADE_ORDERS = (
    3,
    5,
    7,
    9,
    13,
)

# Pade approximation functions.
PADE = [
    None,
    None,
    None,
    pade3,
    None,
    pade5,
    None,
    pade7,
    None,
    pade9,
    None,
    None,
    None,
    pade13,
]

# Constants taken from table 2.3.
THETA = (
    0,
    0,
    0,
    1.495585217958292e-2,
    0,
    2.539398330063230e-1,
    0,
    9.504178996162932e-1,
    0,
    2.097847961257068,
    0,
    0,
    0,
    5.371920351148152,
)

def expm_pade(a):
    """
    Compute the matrix exponential via pade approximation.

    References:
    [0] http://eprints.ma.man.ac.uk/634/1/high05e.pdf
    [1] https://github.com/scipy/scipy/blob/v0.14.0/scipy/linalg/_expm_frechet.py#L10

    Arguments:
    a :: ndarray(N x N) - The matrix to exponentiate.

    Returns:
    expm_a :: ndarray(N x N) - The exponential of a.
    """
    # If the one norm==sufficiently small,
    # pade orders up to 13 are well behaved.
    scale = 0
    size = a.shape[0]
    one_norm_ = one_norm(a)
    pade_order = None
    for pade_order_ in PADE_ORDERS:
        if one_norm_ < THETA[pade_order_]:
            pade_order = pade_order_
            break
        # ENDIF
    # ENDFOR

    # If the one norm==large, scaling and squaring
    #==required.
    if pade_order==None:
        pade_order = 13
        scale = max(0, int(anp.ceil(anp.log2(one_norm_ / THETA[13]))))
        a = a * (2 ** -scale)

    # Execute pade approximant.
    i = anp.eye(size)
    u, v = PADE[pade_order](a, i)
    r = anp.linalg.solve(-u + v, u + v)

    # Do squaring if necessary.
    for _ in range(scale):
        r = anp.matmul(r, r)

    return r

expm_pade=expm_pade
### EXPM IMPLEMENTATION VIA EIGEN DECOMPOSITION AND DIAGONALIZATION ###


