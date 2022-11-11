#!/usr/bin/env python3
"""
module which contains P_init function
"""
import numpy as np


def P_init(X, perplexity):
    """
    initializes all variables required to calculate the P affinities in t-SNE:
    https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf
    - X is a numpy.ndarray of shape (n, d)
      containing the dataset to be transformed by t-SNE
      * n is the number of data points
      * d is the number of dimensions in each point
    - perplexity is the perplexity that all Gaussian distributions should have
    Returns: (D, P, betas, H)
        - D: a numpy.ndarray of shape (n, n) that calculates the squared
          pairwise distance between two data points
        - The diagonal of D should be 0s
        - P: a numpy.ndarray of shape (n, n) initialized to all 0‘s that will
             contain the P affinities
        - betas: a numpy.ndarray of shape (n, 1) initialized to all 1’s that
                 will contain all of the beta values
        - H is the Shannon entropy for perplexity perplexity with a base of 2
    """
    n, _ = X.shape

    # https://nlml.github.io/in-raw-numpy/in-raw-numpy-t-sne/#eq1 for context

    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    np.fill_diagonal(D, 0.)

    P = np.zeros(shape=(n, n))
    betas = np.ones(shape=(n, 1))
    H = np.log2(perplexity)

    return D, P, betas, H
