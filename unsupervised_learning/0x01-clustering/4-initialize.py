#!/usr/bin/env python3
"""
module which contain initialize function
"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    initializes variables for a Gaussian Mixture Model:

    - X is a numpy.ndarray of shape (n, d) containing the data set
    - k is a positive integer containing the number of clusters
    - You are not allowed to use any loops
    Returns: pi, m, S, or None, None, None on failure
      - pi is a numpy.ndarray of shape (k,) containing the priors
        for each cluster, initialized evenly
      - m is a numpy.ndarray of shape (k, d) containing the centroid
        means for each cluster, initialized with K-means
      - S is a numpy.ndarray of shape (k, d, d) containing the covariance
        matrices for each cluster, initialized as identity matrices
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(k, int) or k < 1:
        return None, None, None
    # sum of prior = 1
    pi = np.full((k,), 1 / k)

    # obtaining centroids means for each cluster
    m, _ = kmeans(X, k)

    _, d = X.shape
    # creating S initialized as identity matrices
    S = np.tile(np.identity(d), (k, 1, 1))

    return pi, m, S
