#!/usr/bin/env python3
"""
aadasd
"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    asdasdasdasdasda
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
