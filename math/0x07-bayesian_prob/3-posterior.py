#!/usr/bin/env python3
"""
module which contains posterior function
"""
import numpy as np
likelihood = __import__('0-likelihood').likelihood


def posterior(x, n, P, Pr):
    """
    calculates the posterior probability for the various hypothetical
    probabilities of developing severe side effects given the data:

    - x is the number of patients that develop severe side effects
    - n is the total number of patients observed
    - P is a 1D numpy.ndarray containing the various hypothetical
      probabilities of developing severe side effects
    - Pr is a 1D numpy.ndarray containing the prior beliefs of P
    Returns: the posterior probability of each probability
             in P given x and n, respectively
    """
    if not isinstance(n, int) or n < 1:
        raise ValueError('n must be a positive integer')
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            'x must be an integer that is greater than or equal to 0')
    if x > n:
        raise ValueError('x cannot be greater than n')
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError('P must be a 1D numpy.ndarray')
    if not isinstance(Pr, np.ndarray) or P.shape != Pr.shape:
        raise TypeError('Pr must be a numpy.ndarray with the same shape as P')
    if not np.all((P >= 0) & (P <= 1)):
        raise ValueError('All values in P must be in the range [0, 1]')
    if not np.all((Pr >= 0) & (Pr <= 1)):
        raise ValueError('All values in Pr must be in the range [0, 1]')
    if not np.isclose(Pr.sum(), 1):
        raise ValueError('Pr must sum to 1')
    PBA = likelihood(x, n, P)
    PB = (PBA * Pr).sum()
    PA = Pr
    return (PBA * PA) / PB
