#!/usr/bin/env python3
"""
module which contains rnn function
"""
import numpy as np


def rnn(rnn_cell, X, h_0): 
    """
    performs forward propagation for a simple RNN:

    - rnn_cell is an instance of RNNCell that will
      be used for the forward propagation
    - X is the data to be used, given as a numpy.ndarray of shape (t, m, i)
      * t is the maximum number of time steps
      * m is the batch size
      * i is the dimensionality of the data
    - h_0 is the initial hidden state, as a numpy.ndarray of shape (m, h)
      * h is the dimensionality of the hidden state
    Returns: H, Y
      - H is a numpy.ndarray containing all of the hidden states
      - Y is a numpy.ndarray containing all of the outputs
    """
    h_prev = h_0
    t, _, _ = X.shape
    H, Y = [h_0], []
    for x in X:
        h_prev, y = rnn_cell.forward(h_prev, x)
        H.append(h_prev), Y.append(y)
    return np.array(H), np.array(Y)
