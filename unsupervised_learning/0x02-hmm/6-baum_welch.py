#!/usr/bin/env python3
"""
module which contains baum_welch function
"""
import numpy as np
forward = __import__('3-forward').forward
backward = __import__('5-backward').backward


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """
    performs the Baum-Welch algorithm for a hidden markov model:

    for more info: https://web.stanford.edu/~jurafsky/slp3/A.pdf

    - Observations is a numpy.ndarray of shape (T,)
      that contains the index of the observation
      * T is the number of observations
    - Transition is a numpy.ndarray of shape (M, M)
      that contains the initialized transition probabilities
      * M is the number of hidden states
    - Emission is a numpy.ndarray of shape (M, N)
      that contains the initialized emission probabilities
      * N is the number of output states
    - Initial is a numpy.ndarray of shape (M, 1)
      that contains the initialized starting probabilities
    - iterations is the number of times expectation-maximization
      should be performed
    Returns: the converged Transition, Emission, or None, None on failure
    """
    if not isinstance(Observations, np.ndarray) or len(Observations.shape) != 1:
        return None, None
    T = Observations.shape[0]
    if not isinstance(Emission, np.ndarray) or len(Emission.shape) != 2:
        return None, None
    N, M = Emission.shape
    if not isinstance(Transition, np.ndarray) or len(Transition.shape) != 2:
        return None, None
    if Transition.shape[0] != N or Transition.shape[1] != N:
        return None, None
    if not np.all(Transition.sum(axis=1) == 1):
        return None, None
    if not isinstance(Initial, np.ndarray) or len(Initial.shape) != 2:
        return None, None
    if Initial.shape[0] != N or Initial.shape[1] != 1:
        return None, None
    if Initial.sum() != 1:
        return None, None
    # a = transition b = emmision v = observations 
    for n in range(iterations):
        _, alpha = forward(Observations, Emission, Transition, Initial)
        _, beta = backward(Observations, Emission, Transition, Initial)
        xi = np.zeros(shape=(M, M, T - 1))
        for t in range(T - 1):
            print(alpha.shape, alpha[t, :].T.shape, alpha.T[t, :].shape,Transition.shape)
            denominator = (alpha[t, :].T.dot(Transition) *
                           Emission[:, Observations[t + 1]].T).dot(
                            beta[t + 1, :])
            for s in range(M):
                numerator = alpha[t, s] * Transition[s, :] *\
                            Emission[:, Observations[t + 1]].T *\
                            beta[t + 1, :].T
                xi[s, :, t] = numerator / denominator
        gamma = xi.sum(axis=1)
        Transition = xi.sum(2) / gamma.sum(axis=1).reshape(shape=(-1, 1))

        # Add additional T'th element to gamma
        gamma = np.hstack((gamma, xi[:, :, T - 2].sum(axis=0).reshape(shape=(-1, 1))))

        denominator = gamma.sum(axis=1)

        for l in range(M):
            Emission[:, l] = gamma[:, Observations==l].sum(axis=1)
        
        Emission = Emission.divide(denominator.reshape(axis=(-1, 1)))
    
    return Transition, Emission
