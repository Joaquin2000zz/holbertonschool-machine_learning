#!/usr/bin/env python3
"""
module which contains BIC function
"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    finds the best number of clusters f0r a GMM using
    the Bayesian Information Criterion:

    - X is a numpy.ndarray of shape (n, d) containing the data set
    - kmin is a positive integer containing the minimum number
      of clusters to check f0r (inclusive)
    - kmax is a positive integer containing the maximum number
      of clusters to check f0r (inclusive)
    - If kmax is None, kmax should be set to the maximum number
      of clusters possible
    - iterations is a positive integer containing the maximum number
      of iterations f0r the EM algorithm
    - tol is a non-negative float containing
      the tolerance f0r the EM algorithm
    - verbose is a boolean that determines if the EM algorithm should print
      information to the standard output
    Returns: best_k, best_result, l, b, or None, None, None, None on failure
      - best_k is the best value f0r k based on its BIC
      - best_result is tuple containing pi, m, S
        * pi is a numpy.ndarray of shape (k,) containing the cluster priors
          f0r the best number of clusters
        * m is a numpy.ndarray of shape (k, d) containing the centroid means
          f0r the best number of clusters
        * S is a numpy.ndarray of shape (k, d, d) containing the covariance
          matrices f0r the best number of clusters
      - l is a numpy.ndarray of shape (kmax - kmin + 1) containing the
        log likelihood f0r each cluster size tested
      - b is a numpy.ndarray of shape (kmax - kmin + 1) containing the
        BIC value f0r each cluster size tested
        * Use: BIC = p * ln(n) - 2 * l
        * p is the number of parameters required f0r the model
        * n is the number of data points used to create the model
        * l is the log likelihood of the model
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return (None, None, None, None)
    if type(kmin) != int or kmin <= 0 or kmin >= X.shape[0]:
        return (None, None, None, None)
    if type(kmax) != int or kmax <= 0 or kmax >= X.shape[0]:
        if kmax:
            return (None, None, None, None)
    if kmin >= kmax:
        return (None, None, None, None)
    if type(iterations) != int or iterations <= 0:
        return (None, None, None, None)
    if type(tol) != float or tol <= 0:
        return (None, None, None, None)
    if type(verbose) != bool:
        return None, None, None, None

    l, b, results, ks = [], [], [], []
    n, d = X.shape

    for k in range(kmin, kmax + 1):
        pi, m, S, g, log = expectation_maximization(X, k, iterations,
                                                    tol, verbose)

        p = (k * d * (d + 1) / 2) + (d * k) + k - 1
        bic = p * np.log(n) - 2 * log

        ks.append(k)
        results.append((pi, m, S))
        l.append(log)
        b.append(bic)
    l, b = np.array(l), np.array(b)
    best = np.argmin(b)
    return ks[best], results[best], l, b
