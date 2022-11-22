#!/usr/bin/env python3
"""
module which contains regular function
"""
import numpy as np


def regular(P):
    """
    determines the steady state probabilities of a regular markov chain:

    * P is a is a square 2D numpy.ndarray of shape (n, n)
      representing the transition matrix
    * P[i, j] is the probability of transitioning from state i to state j
    * n is the number of states in the markov chain
    Returns: a numpy.ndarray of shape (1, n) containing the steady
             state probabilities, or None on failure
    """
    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return None
    if P.shape[0] != P.shape[1]:
        return None
    if np.any(P.sum(axis=1) != 1):
        return None

    n, n = P.shape

    evals, evecs = np.linalg.eig(P.T)
    evec1 = evecs[:, np.isclose(evals, 1)]

    try:
        evec1 = evec1[:, 0]
    except Exception:
        return None

    S = evec1 / evec1.sum()

    if 0 in S:
        return None

    return np.expand_dims(S, axis=0) if np.allclose(S @ P, S) else None
