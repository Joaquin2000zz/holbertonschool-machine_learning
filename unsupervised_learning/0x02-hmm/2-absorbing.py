#!/usr/bin/env python3
"""
module which contains absorbing function
"""
import numpy as np


def check_columns(T, k, first=True):
    """
    check if that column has an absobing chain
    - T is the transpose of the standar transition matrix
    - k is the position in the diagonal to check
    - first is a flag to determine whether is the first recursion or not
    Return True if it is absorbing, or False on failure
    """
    n, n = T.shape
    if np.any(T[k][k + 1:] != 0):
        if k < n and first:
            return check_columns(T, k + 1, False)
        return True

    return False


def absorbing(P):
    """
    determines if a markov chain is absorbing:

    - P is a is a square 2D numpy.ndarray of shape (n, n) representing
      the standard transition matrix
    - P[i, j] is the probability of transitioning from state i to state j
    - n is the number of states in the markov chain
    Returns: True if it is absorbing, or False on failure
    """
    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return False
    if P.shape[0] != P.shape[1]:
        return False
    if np.any(P.sum(axis=1) != 1):
        return False

    D = P.diagonal()
    if np.all(D == 1):
        return True

    pos = np.where(D == 1)[0]

    for i in pos:
        if check_columns(P.T, i):
            return True
    return False
