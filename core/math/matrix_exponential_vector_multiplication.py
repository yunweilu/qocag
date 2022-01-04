"""
Action of matrix exponential and action of propagator derivative on a vector or a set of basis
"""

import numpy as np
import scipy.sparse.linalg
from scipy.sparse import bmat, isspmatrix
import jax.numpy as jnp

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
    state = np.complex128(state)
    if tol == None:
        tol = 2 ** -53
    HILBERT_SIZE = state.shape[0]
    if isspmatrix(A) is False:
        c = np.block([[A, E], [np.zeros_like(A), A]])
    else:
        c = bmat([[A, E], [None, A]]).tocsc()
    state = state.flatten()
    state0 = np.zeros_like(state)
    state = np.block([state0, state])
    state = expmat_vec_mul(c, state, tol)
    new = state[HILBERT_SIZE:2 * HILBERT_SIZE]
    state = state[0:HILBERT_SIZE]

    return state.reshape((HILBERT_SIZE, 1)), new.reshape((HILBERT_SIZE, 1))


"""Compute the action of the matrix exponential."""


def one_norm(A):
    """
    Args:
    A :: numpy.ndarray - matrix or vector
    Returns:
    one norm of A
    """
    if scipy.sparse.isspmatrix(A):
        return max(abs(A).sum(axis=0).flat)
    else:
        return np.linalg.norm(A, 1)

def one_norm_ad(A):
    """
    Used for jax.
    Args:
    A :: numpy.ndarray - matrix or vector
    Returns:
    one norm of A
    """

    return jnp.linalg.norm(A, 1)

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

def trace_ad(A):
    """
    Used for jax.
    Args:
    A :: numpy.ndarray - matrix
    Returns:
    trace of A
    """
    return jnp.trace(A)


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

def ident_like_ad(A):
    """
    Used for jax.
    Args:
    A :: numpy.ndarray - matrix
    Returns:
    Identity matrix which has same dimension as A
    """
    return jnp.eye(A.shape[0], A.shape[1], dtype=A.dtype)

def get_s(A,tol):
    """
        Determine s in scaling and squaring.
        Args:
        A :: numpy.ndarray - matrix
        tol :: numpy.float64 - expected error
        Returns:
        s :: int
    """
    s = 1
    while (1):
        tol_power = np.ceil(np.log10(tol))
        norm_A = one_norm(A) / s
        max_term_notation = np.floor(norm_A)
        max_term = 1
        for i in range(1, np.int64(max_term_notation)):
            max_term = max_term * norm_A / i
            max_power = np.ceil(np.log10(max_term))
            if max_power > 30:
                break
        max_power = np.ceil(np.log10(max_term))
        if max_power - 16 <= tol_power:
            break
        s = s + 1
    return s

def get_s_ad(A,tol):
    """
    Determine s in scaling and squaring. Used for jax.
    Args:
    A :: numpy.ndarray - matrix
    tol :: numpy.float64 - expected error
    Returns:
    s :: int
    """
    s = 1
    while (1):
        tol_power = jnp.ceil(jnp.log10(tol))
        norm_A = one_norm(A) / s
        max_term_notation = jnp.floor(norm_A)
        max_term = 1
        for i in range(1, jnp.int64(max_term_notation)):
            max_term = max_term * norm_A / i
            max_power = jnp.ceil(jnp.log10(max_term))
            if max_power > 30:
                break
        max_power = jnp.ceil(jnp.log10(max_term))
        if max_power - 16 <= tol_power:
            break
        s = s + 1
    return s



def expmat_vec_mul(A, b, tol=None):
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
    if tol is None:
        tol = 1e-16
    s = get_s(A,tol)
    f = b
    j = 0
    while (1):
        eta = np.exp(mu / float(s))
        coeff = s * (j + 1)
        b = A.dot(b) / coeff
        c2 = one_norm(b)
        f = f + b
        total_norm = one_norm(f)
        if c2 / total_norm < tol:
            m = j + 1
            break
        j = j + 1
    f = eta * f
    b = f
    for i in range(1, int(s)):
        eta = np.exp(mu / float(s))
        for j in range(m):
            coeff = s * (j + 1)
            b = A.dot(b) / coeff
            c2 = one_norm(b)
            f = f+ b
            total_norm = one_norm(f)
            if c2 / total_norm < tol:
                m = j + 1
                break
        f = eta * f
        b = f
    return f


def expmat_vec_mul_ad(A, b, tol=None):
    """
    Compute the exponential matrix and vector multiplication e^(A) B. Used for jax.
    Args:
    A :: numpy.ndarray - matrix
    b :: numpy.ndarray - vector or a set of basis vectors
    tol :: numpy.float64 - expected error
    Returns:
    f :: numpy.ndarray - Approximation of e^(A) b
    """
    ident = ident_like_ad(A)
    n = A.shape[0]
    mu = trace_ad(A) / float(n)
    # Why mu? http://eprints.ma.man.ac.uk/1591/, section 3.1
    A = A - mu * ident
    if tol is None:
        tol = 1e-16
    s = get_s_ad(A,tol)
    f = b
    j = 0
    while (1):
        eta = jnp.exp(mu / float(s))
        coeff = s * (j + 1)
        b = jnp.matmul(A,b) / coeff
        c2 = one_norm_ad(b)
        f = f + b
        total_norm = one_norm_ad(f)
        if c2 / total_norm < tol:
            m = j + 1
            break
        j = j + 1
    f = eta * f
    b = f
    for i in range(1, int(s)):
        eta = jnp.exp(mu / float(s))
        for j in range(m):
            coeff = s * (j + 1)
            b = jnp.matmul(A,b)  / coeff
            c2 = one_norm_ad(b)
            f = f+ b
            total_norm = one_norm_ad(f)
            if c2 / total_norm < tol:
                m = j + 1
                break
        f = eta * f
        b = f
    return f

