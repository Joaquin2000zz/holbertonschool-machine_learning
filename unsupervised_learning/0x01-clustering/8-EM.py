#!/usr/bin/env python3
"""
module which contains expectation-maximization function
"""
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    performs the expectation maximization f0r a GMM:

    - X is a numpy.ndarray of shape (n, d) containing the data set
    - k is a positive integer containing the number of clusters
    - iterations is a positive integer containing the maximum number
      of iterations f0r the algorithm
    - tol is a non-negative float containing tolerance of the log likelihood,
      used to determine early stopping i.e. if the difference is less
      than or equal to tol you should stop the algorithm
    - verbose is a boolean that determines if you should print information
      about the algorithm
    Returns: pi, m, S, g, l, or None, None, None, None, None on failure
      - pi is a numpy.ndarray of shape (k,)
        containing the priors f0r each cluster
      - m is a numpy.ndarray of shape (k, d) containing the centroid means
        f0r each cluster
      - S is a numpy.ndarray of shape (k, d, d) containing the covariance
        matrices to each cluster
      - g is a numpy.ndarray of shape (k, n) containing the probabilities
        to each data point in each cluster
      - l is the log likelihood of the model
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k < 1 or k > X.shape[0]:
        return None, None, None, None, None
    if not isinstance(iterations, int) or iterations < 1:
        return None, None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None, None

    # initializing variables of Gaussian Mixture model
    # i.e. prior, mean and Covariance
    pi, m, S = initialize(X, k)

    l0 = .0

    for i in range(iterations):
        # computing expectation. returns:
        # - g which are the posterior probabilities
        #   of each datapoint in each cluster
        # - l is the total log likelihood
        #   i.e. how well the model fits the data
        g, l1 = expectation(X, pi, m, S)
        if verbose and i % 10 == 0:
            msg = 'Log Likelihood after {} iterations: {}'.format(i,
                                                                  l1.round(5))
            print(msg)
        # if the difference is less than or equal
        # the algorithm is stopped
        if i > 0 and np.abs(l0 - l1) <= tol:
            if verbose:
                xd = l1.round(5)
                msg = 'Log Likelihood after {} iterations: {}'.format(i, xd)
                print(msg)
            return pi, m, S, g, l1
        # computing maximitation. returns:
        # new prior, mean and covariance of each cluster
        pi, m, S = maximization(X, g)

        l0 = l1

    g, l1 = expectation(X, pi, m, S)
    if verbose:
        msg = 'Log Likelihood after {} iterations: {}'.format(i + 1,
                                                              l1.round(5))
        print(msg)
    return pi, m, S, g, l1
