#!/usr/bin/env python3
"""
module which contains entropy function
"""
import numpy as np


def HP(Di, beta):
    """
    calculates the Shannon entropy and P affinities relative to a data point:
    https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf

    - Di is a numpy.ndarray of shape (n - 1,) containing the
      pariwise distances between a data point and
      all other points except itself
        * n is the number of data points
    - beta is a numpy.ndarray of shape (1,) containing the beta value
      for the Gaussian distribution
    Returns: (Hi, Pi)
      - Hi: the Shannon entropy of the points
      - Pi: a numpy.ndarray of shape (n - 1,) containing the P
        affinities of the points
      - Hint: see page 4 of t-SNE
    """
    # https://nlml.github.io/in-raw-numpy/in-raw-numpy-t-sne/#eq1 for context
    P = np.exp(-Di * beta)
    Pi = P / np.sum(P)

    Hi = -np.sum(Pi * np.log2(Pi))

    return Hi, Pi
