#!/usr/bin/env python3
"""
module which contains MultiNormal class
"""
import numpy as np


class MultiNormal:
    """
    represents a multivariate normal distribution
    """

    def __init__(self, data):
        """
        class constructor
        - data is a numpy.ndarray of shape (d, n) containing the data set:
          * n is the number of data points
          * d is the number of dimensions in each data point
        - Set the public instance variables:
          * mean - a numpy.ndarray of shape (d, 1)
                   containing the mean of data
          * cov - a numpy.ndarray of shape (d, d)
                  containing the covariance matrix data
        """
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError('data must be a 2D numpy.ndarray')
        _, n = data.shape
        if n < 2:
            raise ValueError('data must contain multiple data points')

        self.mean = np.mean(data, axis=1, keepdims=True)
        Xμ = data - self.mean
        self.cov = (Xμ @ Xμ.T) / (n - 1)

    def pdf(self, x):
        """
        calculates the PDF at a data point:
        - x is a numpy.ndarray of shape (d, 1)
          containing the data point whose PDF should be calculated
            * d is the number of dimensions of the MultiNormal instance
        Returns the value of the PDF
        """
        if not isinstance(x, np.ndarray):
            raise TypeError('x must be a numpy.ndarray')
        error = f'x must have the shape ({self.mean.shape[0]}, 1)'
        if len(x.shape) != 2:
            raise ValueError(error)
        d, h = x.shape
        if d != self.mean.shape[0] and h != 1:
            raise ValueError(error)
        det = np.linalg.det(self.cov)
        inv = np.linalg.inv(self.cov)
        μ = self.mean
        Xμ = x - μ
        coefficient = 1 / np.sqrt(((2 * np.pi) ** d) * det)
        exp = np.exp(-(np.dot(np.dot(Xμ.T, inv), Xμ)) / 2)
        return np.sum(coefficient * exp)
