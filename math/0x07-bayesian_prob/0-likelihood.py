#!/usr/bin/env python3
"""
module which contains likelihood function

You are conducting a study on a revolutionary cancer drug and are looking
to find the probability that a patient who takes this drug will develop
severe side effects. During your trials, n patients take the drug and
x patients develop severe side effects.
You can assume that x follows a binomial distribution.
"""
import numpy as np


def likelihood(x, n, P):
    """
    calculates the likelihood of obtaining this data given various
    hypothetical probabilities of developing severe side effects:

    - x is the number of patients that develop severe side effects
    - n is the total number of patients observed
    - P is a 1D numpy.ndarray containing the various hypothetical
      probabilities of developing severe side effects
    Returns: a 1D numpy.ndarray containing the likelihood
             of obtaining the data, x and n,
             for each probability in P, respectively
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
    if not np.all((P >= 0) & (P <= 1)):
        raise ValueError('All values in P must be in the range [0, 1]')

    f = np.math.factorial
    # using formula of binomial distribution
    comb = f(n) / (f(n - x) * f(x))
    success = np.power(P, x)
    failure = np.power(1 - P, n - x)
    return comb * success * failure
