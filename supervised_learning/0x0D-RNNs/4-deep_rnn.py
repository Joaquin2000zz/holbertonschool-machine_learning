#!/usr/bin/env python3
"""
module which contains deep_rnn function
"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    performs forward propagation for a deep RNN:

    - rnn_cells is a list of RNNCell instances of length l that will be used
      for the forward propagation
      * l is the number of layers
    - X is the data to be used, given as a numpy.ndarray of shape (t, m, i)
      * t is the maximum number of time steps
      * m is the batch size
      * i is the dimensionality of the data
    - h_0 is the initial hidden state, as a np.ndarray of shape (l, m, h)
      * h is the dimensionality of the hidden state
    Returns: H, Y
    - H is a numpy.ndarray containing all of the hidden states
    - Y is a numpy.ndarray containing all of the outputs
    """
    T, M, I = X.shape
    L, M, H = h_0.shape

    H = np.zeros(shape=(T + 1, L, M, H))
    H[0] = h_0
    print(T, H.shape)
    for t in range(T):
        x = X[t]
        for l in range(L):
            x, yl = rnn_cells[l].forward(H[t, l], x)
            H[t + 1, l] = x
            if l == L - 1:
                if t == 0:
                    h, w = yl.shape
                    Y = np.zeros(shape=(T, h, w))
                    Y[t] = yl[np.newaxis]
                else:
                    Y[t] = yl[np.newaxis]
    return H, Y
