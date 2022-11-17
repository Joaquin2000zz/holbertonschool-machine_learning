#!/usr/bin/env python3
"""
module which contains maximization function
"""
import numpy as np


def maximization(X, g):
    """
    calculates the maximization step in the EM algorithm for a GMM:
    https://brilliant.org/wiki/gaussian-mixture-model/
    - X is a numpy.ndarray of shape (n, d) containing the data set
    - g is a numpy.ndarray of shape (k, n) containing the posterior
      probabilities for each data point in each cluster
    - You may use at most 1 loop
    Returns: pi, m, S, or None, None, None on failure
      - pi is a numpy.ndarray of shape (k,) containing the updated
        priors to each cluster
      - m is a numpy.ndarray of shape (k, d) containing the updated
        centroid means for each cluster
      - S is a numpy.ndarray of shape (k, d, d) containing the updated
        covariance matrices for each cluster
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None
    if not (X.shape[0] == g.shape[1]):
        return None, None, None
    p = g.sum(axis=0)
    if not (p.all() == 1) or p.sum() != X.shape[0]:
        return None, None, None

    _, d = X.shape
    k, n = g.shape
    sigma_g = g.sum(axis=1)
    pi = sigma_g / n
    m = np.zeros(shape=(k, d))
    S = np.zeros(shape=(k, d, d))

    for i in range(k):
        pi[i] = (g[i] / n).sum()
        m[i] = g[i] @ X / sigma_g[i]
        Xµ = X - m[i]
        S[i] = g[i] * Xµ.T @ Xµ / sigma_g[i]
    return pi, m, S
