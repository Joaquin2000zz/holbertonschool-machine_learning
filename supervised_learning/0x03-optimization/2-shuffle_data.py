#!/usr/bin/env python3
"""
module which contains shuffle_data function
"""
import numpy as np


def shuffle_data(X, Y):
    """
    Shuffles the data points in two matrices the same way:
    * X is the first numpy.ndarray of shape (m, nx) to shuffle
        - m is the number of data points
        - nx is the number of features in X
    * Y is the second numpy.ndarray of shape (m, ny) to shuffle
        - m is the same number of data points as in X
        - ny is the number of features in Y
    * random.permutation(x)
        Randomly permute a sequence, or return a permuted range.
        If x is a multi-dimensional array,
        it is only shuffled along its first index.
    Returns: the shuffled X and Y matrices
    """
    r_permuted = np.random.permutation(X.shape[0])
    return X[r_permuted], Y[r_permuted]
