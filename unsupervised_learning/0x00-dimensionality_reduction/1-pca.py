#!/usr/bin/env python3
"""
module which contains pca function
"""
import numpy as np


def pca(X, ndim):
    """
    given a dataset, peforms its pca
    - X is a numpy.ndarray of shape (n, d) where:
      * n is the number of data points
      * d is the number of dimensions in each point
      * all dimensions have a mean of 0 across all data points
    - ndim is the new dimensionality of the transformed X
    Returns: the weights matrix, W, that maintains
             var fraction of X‘s original variance
    - W is a numpy.ndarray of shape (d, nd) where nd is
      the new dimensionality of the transformed X
    ### this is an unoptimized implementation using the
        eigen vectors of the covariance matrix ###
    # mean centering the data
    Xµ = X - np.mean(X, axis=0)

    # computing covariance of mean centered data
    cov = np.cov(Xμ, rowvar=False)

    # computing both, eigen values and vectors of covariance matrix
    eigen_val, eigen_vec = np.linalg.eigh(cov)

    # sorting eigen vector in descending order
    idx = np.argsort(eigen_val)[::-1]
    eigen_vec = eigen_vec[..., idx]

    # selecting first n eigenvectors
    eigen_vec = eigen_vec[..., :ndim]

    # returning reduced data
    return Xμ @ eigen_vec
    """
    # using singular value decomposition technique
    # to compute principal component analysis

    Xµ = X - np.mean(X, axis=0)

    U, sigma, V = np.linalg.svd(Xμ)

    return Xμ @ V[:ndim].T
