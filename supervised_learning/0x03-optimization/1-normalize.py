#!/usr/bin/env python3
"""
module which contains normalize function
"""


def normalize(X, m, s):
    """
    normalizes (standardizes) a matrix:

    * X is the numpy.ndarray of shape (d, nx) to normalize
        - d is the number of data points
        - nx is the number of features
    * m is a numpy.ndarray of shape (nx,) that contains
      the mean of all features of X
    * s is a numpy.ndarray of shape (nx,) that contains
      the standard deviation of all features of X
    Returns: The normalized X matrix
    """
    Z = (X - m) / s
    return Z
"""
usar np.ex y una matriz identidad para crear una clase que dibuje el plano 3d de elevar e a dicha matriz
"""