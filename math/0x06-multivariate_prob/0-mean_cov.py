#!/usr/bin/env python3
"""
module which contains mean_cov function
"""
import numpy as np


def mean_cov(X):
    """
    given a matrix, calculates its mean and covariance
    - X is a numpy.ndarray of shape (n, d) containing the data set:
      * n is the number of data points
      * d is the number of dimensions in each data point
    Returns: μ, cov:
      - μ is a numpy.ndarray of shape (1, d)
        containing the mean of the data set
      - cov is a numpy.ndarray of shape (d, d)
        containing the covariance matrix of the data set
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        raise TypeError('X must be a 2D numpy.ndarray')
    n, _ = X.shape
    if n < 2:
        raise ValueError('X must contain multiple data points')

    μ = np.mean(X, axis=0, keepdims=True)

    Xμ = X - μ

    return μ, (Xμ.T @ Xμ) / (n - 1)
