#!/usr/bin/env python3
"""
module which contains foward function
"""
import numpy as np

def forward(Observation, Emission, Transition, Initial):
    """
    performs the forward algorithm for a hidden markov model:

    for more info: https://web.stanford.edu/~jurafsky/slp3/A.pdf

    - Observation is a numpy.ndarray of shape (T,)
      that contains the index of the observation
      * T is the number of observations
    - Emission is a numpy.ndarray of shape (N, M) containing the
      emission probability of a specific observation
      given a hidden state
      * Emission[i, j] is the probability of observing j
        given the hidden state i
      * N is the number of hidden states
      * M is the number of all possible observations
    - Transition is a 2D numpy.ndarray of shape (N, N)
      containing the transition probabilities
      * Transition[i, j] is the probability of transitioning
        from the hidden state i to j
    - Initial a numpy.ndarray of shape (N, 1) containing the probability
      of starting in a particular hidden state
    Returns: P, F, or None, None on failure
    - P is the likelihood of the observations given the model
    - F is a numpy.ndarray of shape (N, T) containing the
      forward path probabilities
    - F[i, j] is the probability of being in hidden state i at time j given
      the previous observations
    """
    if not isinstance(Observation, np.ndarray) or len(Observation.shape) != 1:
        print('entre al if 1')
        return None, None
    T = Observation.shape[0]
    if not isinstance(Emission, np.ndarray) or len(Emission.shape) != 2:
        print('entre al if 2')
        return None, None
    N, M = Emission.shape
    if not isinstance(Transition, np.ndarray) or len(Transition.shape) != 2:
        print('entre al if 3')
        return None, None
    if Transition.shape[0] != N or Transition.shape[1] != N:
        print(f'entre al if 4 transition shape {Transition.shape} n {N}')
        return None, None
    if not np.all(Transition.sum(axis=1) == 1):
        print('entre al if 5')
        return None, None
    if not isinstance(Initial, np.ndarray) or len(Initial.shape) != 2:
        print('entre al if 6')
        return None, None
    if Initial.shape[0] != N or Initial.shape[1] != 1:
        print('entre al if 7')
        return None, None
    if Initial.sum() != 1:
        print('entre al if 8')
        return None, None

    F = np.zeros(shape=(N, T))

    # initialization step vectorized
    F[:, 0] = Initial.T * Emission[:, Observation[0]]

    for t in range(1, T):  # recursion step
        for s in range(N):
            F[s, t] = (F[:, t - 1] * Transition[:, s] *
                       Emission[s, Observation[t]]).sum()

    return F[:, T - 1].sum(), F
