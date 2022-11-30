#!/usr/bin/env python3
"""
module which contains GaussianProcess's class
"""
import numpy as np


class GaussianProcess:
    """
    represents a noiseless 1D Gaussian process
    """

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        - X_init is a numpy.ndarray of shape (t, 1)
          representing the inputs already sampled with the black-box function
        - Y_init is a numpy.ndarray of shape (t, 1)
          representing the outputs of the black-box function
          for each input in X_init
        - t is the number of initial samples
        - l is the length parameter for the kernel
        - sigma_f is the standard deviation given to the output of the
          black-box function
        - Sets the public instance attributes X, Y, l, and sigma_f
          corresponding to the respective constructor inputs
        - Sets the public instance attribute K, representing the current
          covariance kernel matrix for the Gaussian process
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        """
        calculates the covariance kernel matrix between two matrices:

        - X1 is a numpy.ndarray of shape (m, 1)
        - X2 is a numpy.ndarray of shape (n, 1)
        - the kernel should use the Radial Basis Function (RBF)
        Returns: the covariance kernel matrix
                 as a numpy.ndarray of shape (m, n)
        """
        sigma_X1 = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
        sigma_X2 = np.sum(X2 ** 2, axis=1)
        sqdist = sigma_X1 + sigma_X2 - 2 * X1.dot(X2.T)
        return self.sigma_f ** 2 * np.exp(-.5 / self.l ** 2 * sqdist)

    def predict(self, X_s):
        """
        predicts the mean and standard deviation of points
        in a Gaussian process:
        - X_s is a numpy.ndarray of shape (s, 1) containing all of the points
          whose mean and standard deviation should be calculated
        - s is the number of sample points
        Returns: mu, sigma
          - mu is a numpy.ndarray of shape (s,) containing the mean
            for each point in X_s, respectively
          - sigma is a numpy.ndarray of shape (s,) containing the variance
            for each point in X_s, respectively
        """
        t = self.X.shape[0]
        K = self.kernel(self.X, self.X) + 1e-8 ** 2 * np.eye(t)
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s) + 1e-8 * np.eye(len(X_s))
        K_inv = np.linalg.inv(K)

        mu = K_s.T.dot(K_inv).dot(self.Y).squeeze(axis=1)

        cov = K_ss - K_s.T.dot(K_inv).dot(K_s)

        return mu, np.diag(cov)

    def update(self, X_new, Y_new):
        """
        Updates a Gaussian Process:
        - X_new is a numpy.ndarray of shape (1,)
          that represents the new sample point
        - Y_new is a numpy.ndarray of shape (1,)
          that represents the new sample function value
        - Updates the public instance attributes X, Y, and K
        """
        self.X = np.concatenate((self.X, X_new[np.newaxis]), axis=0)
        self.Y = np.concatenate((self.Y, Y_new[np.newaxis]), axis=0)
        self.K = self.kernel(self.X, self.X)
