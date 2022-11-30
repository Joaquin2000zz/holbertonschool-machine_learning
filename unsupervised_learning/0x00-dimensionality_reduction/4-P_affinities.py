#!/usr/bin/env python3
"""
module which contains P_affinities function
"""
import numpy as np
P_init = __import__('2-P_init').P_init
HP = __import__('3-entropy').HP


def P_affinities(X, tol=1e-5, perplexity=30.0):
    """
    calculates the symmetric P affinities of a data set:
    https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf

    - X is a numpy.ndarray of shape (n, d) containing the
      dataset to be transformed by t-SNE
        * n is the number of data points
        * d is the number of dimensions in each point
    - tol is the maximum tolerance allowed (inclusive) for the
      difference in Shannon
      entropy from perplexity for all Gaussian distributions
    - perplexity is the perplexity that all Gaussian distributions should have
    Returns: P, a numpy.ndarray of shape (n, n)
             containing the symmetric P affinities
    Hint 1: See page 6 of t-SNE
    Hint 2: For this task, you will need to perform a binary search
            on each point’s distribution to find the correct value of beta
            that will give a Shannon Entropy H within the tolerance
            (Think about why we analyze the Shannon
            entropy instead of perplexity). Since beta can be in the range
            (0, inf), you will have to do a binary search with the high
            and low initially set to None. If in your search, you are supposed
            to increase/decrease beta to high/low but they are still set to None,
            you should double/half the value of beta instead.
    """
    D, P, beta, logU = P_init(X, perplexity)

    n, d = X.shape
    # Loop over all datapoints
    # Loop over all datapoints
    (n, d) = X.shape
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)
    for i in range(n):
        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point ", i, " of ", n, "...")
        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        (H, thisP) = HP(Di, beta[i])
        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:
            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i]
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2
                else:
                    beta[i] = (beta[i] + betamax) / 2
            else:
                betamax = beta[i]
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2
                else:
                    beta[i] = (beta[i] + betamin) / 2
            # Recompute the values
            (H, thisP) = HP(Di, beta[i])
            Hdiff = H - logU
            tries = tries + 1
        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP
        # Return final P-matrix
    print("Mean value of sigma: ", np.mean(np.sqrt(1 / beta)))
    return P
