#!/usr/bin/env python3
"""
module which contains correlation function
"""
import numpy as np


def correlation(C):
    """
    given a covariance matrix, computes its correlation
    - C is a numpy.ndarray of shape (d, d) containing a covariance matrix
      * d is the number of dimensions
    Returns a numpy.ndarray of shape (d, d) containing the correlation matrix
    """
    if not isinstance(C, np.ndarray) or len(C.shape) != 2:
        raise TypeError('C must be a numpy.ndarray')
    w, h = C.shape
    if w != h:
        raise ValueError('C must be a 2D square matrix')
    c = w
    var = np.diag(C)
    corr = np.zeros(shape=(c, c))
    for i in range(c):
        for j in range(c):
            if j > i:
                break
            corr[i][j] = corr[j][i] = C[i][j] / np.sqrt(var[i] * var[j])
    return corr
