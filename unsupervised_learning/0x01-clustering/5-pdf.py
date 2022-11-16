#!/usr/bin/env python3
"""
module which contains pdf function
"""
import numpy as np


def pdf(X, m, S):
    """
    calculates the probability density function of a Gaussian distribution:
    # https://brilliant.org/wiki/gaussian-mixture-model/

    - X is a numpy.ndarray of shape (n, d) containing the data points
      whose PDF should be evaluated
    - m is a numpy.ndarray of shape (d,) containing
      the mean of the distribution
    - S is a numpy.ndarray of shape (d, d) containing
      the covariance of the distribution
    - You are not allowed to use any loops
    - You are not allowed to use the function numpy.diag or
      the method numpy.ndarray.diagonal
    Returns: P, or None on failure
      - P is a numpy.ndarray of shape (n,) containing the PDF values
        for each data point
      - All values in P should have a minimum value of 1e-300
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(m, np.ndarray) or len(m.shape) != 1:
        return None
    if not isinstance(S, np.ndarray) or len(S.shape) != 2:
        return None
    if not (X.shape[1] == m.shape[0] == S.shape[0] == S.shape[1]):
        return None

    n, d = X.shape
    det = np.linalg.det(S)
    inv = np.linalg.inv(S)
    Xµ = X - m

    factor = 1 / np.sqrt((2 * np.pi) ** (d) * det)
    exp = np.exp(-0.5 * np.sum(Xμ * np.matmul(Xμ, inv), axis=1))

    PDF = factor * exp
    return np.where(PDF < 1e-300, 1e-300, PDF)
