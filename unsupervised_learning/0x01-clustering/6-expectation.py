#!/usr/bin/env python3
"""
module which contains expectation function
"""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    calculates the expectation step in the EM algorithm for a GMM:
    https://brilliant.org/wiki/gaussian-mixture-model/
    - X is a numpy.ndarray of shape (n, d) containing the data set
    - pi is a numpy.ndarray of shape (k,) containing the
      priors for each cluster
    - m is a numpy.ndarray of shape (k, d) containing the
      centroid means for each cluster
    - S is a numpy.ndarray of shape (k, d, d) containing the covariance
      matrices for each cluster
    - You may use at most 1 loop
    Returns: g, l, or None, None on failure
      - g is a numpy.ndarray of shape (k, n) containing the posterior
        probabilities for each data point in each cluster
      - l is the total log likelihood
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(pi, np.ndarray) or len(pi.shape) != 1:
        return None
    if not isinstance(m, np.ndarray) or len(m.shape) != 2:
        return None
    if not isinstance(S, np.ndarray) or len(S.shape) != 3:
        return None
    if not (X.shape[1] == m.shape[1] == S.shape[1] == S.shape[2]):
        return None
    if not (pi.shape[0] == m.shape[0] == S.shape[0]):
        return None

    k = pi.shape[0]
    n = X.shape[0]
    P = np.zeros(shape=(k, n))
    l = []
    for i in range(k):
        # ith posterior probability it's the ith prior times the ith pdf
        P[i] = pi[i] * pdf(X, m[i], S[i])

    # adding up the stacked posterior probabilities
    p = P.sum(axis=0)

    g = P / p
    return g, np.log(p).sum()
