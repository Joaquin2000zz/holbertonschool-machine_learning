#!/usr/bin/env python3
"""
module which contains variance function
"""
import numpy as np


def variance(X, C):
    """
    calculates the total intra-cluster variance for a data set:

    - X is a numpy.ndarray of shape (n, d) containing the data set
    - C is a numpy.ndarray of shape (k, d) containing
      the centroid means for each cluster
    - You are not allowed to use any loops
    Returns: var, or None on failure
      - var is the total variance
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(C, np.ndarray) or len(C.shape) != 2:
        return None
    if X.shape[1] != C.shape[1]:
        return None
    # euclidean distances in between the X data points and C centroid points
    distances = np.linalg.norm(X - C[:, np.newaxis], axis=-1)
    # minimum distances in between X and C
    minDis = distances.min(axis=0)
    # as https://www.youtube.com/watch?v=9W45yMcAOsw says
    # the variance is the sum of all the squared minimum distances
    return (minDis ** 2).sum()
