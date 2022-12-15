#!/usr/bin/env python3
"""
module which contains bi_rnn function
"""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    performs forward propagation for a bidirectional RNN:

    - bi_cell is an instance of BidirectinalCell that will be used
      for the forward propagation
    - X is the data to be used, given as a numpy.ndarray of shape (t, m, i)
      * t is the maximum number of time steps
      * m is the batch size
      * i is the dimensionality of the data
    - h_0 is the initial hidden state in the forward direction, given as
      a numpy.ndarray of shape (m, h)
      * h is the dimensionality of the hidden state
    - h_t is the initial hidden state in the backward direction,
      given as a numpy.ndarray of shape (m, h)
    Returns: H, Y
    -  H is a numpy.ndarray containing all of the concatenated hidden states
    -  Y is a numpy.ndarray containing all of the outputs
    """

    T, M, _ = X.shape
    _, H = h_0.shape
    H_f, H_b = np.zeros(shape=(T, M, H)), np.zeros(shape=(T, M, H))

    for t in range(T):
        x_f, x_b = X[t], X[-(t + 1)]

        h_0, h_t = bi_cell.forward(h_0, x_f), bi_cell.backward(h_t, x_b)

        H_f[t], H_b[-(t + 1)] = h_0, h_t

    H = np.concatenate((H_f, H_b), axis=-1)
    Y = bi_cell.output(H)

    return H, Y
