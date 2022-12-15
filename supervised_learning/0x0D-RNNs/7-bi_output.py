#!/usr/bin/env python3
"""
module which contains BidirectionalCell class
"""
import numpy as np


class BidirectionalCell:
    """
    represents a bidirectional cell of an RNN
    """

    def __init__(self, i, h, o):
        """
        class constructor

        - i is the dimensionality of the data
        - h is the dimensionality of the hidden states
        - o is the dimensionality of the outputs
        - Creates the public instance attributes Whf, Whb, Wy, bhf, bhb, by
          that represent the weights and biases of the cell
          * Whf and bhf are for the hidden states in the forward direction
          * Whb and bhb are for the hidden states in the backward direction
          * Wy and by are for the outputs
        - The weights should be initialized using a random normal distribution
          in the order listed above
        - The weights will be used on the right side for matrix multiplication
        - The biases should be initialized as zeros
        """
        self.Whf = np.random.normal(size=(i + h, h))
        self.bhf = np.zeros(shape=(1, h))
        self.Whb = np.random.normal(size=(i + h, h))
        self.bhb = np.zeros(shape=(1, h))
        self.Wy = np.random.normal(size=(i + h + o, o))
        self.by = np.zeros(shape=(1, o))

    def forward(self, h_prev, x_t):
        """
        - that calculates the hidden state in the forward direction for
          one time step
        - x_t is a numpy.ndarray of shape (m, i) that contains the data
          input for the cell
          * m is the batch size for the data
        - h_prev is a numpy.ndarray of shape (m, h) containing the previous
          hidden state
        Returns: h_next, the next hidden state
        """
        Whh = np.concatenate((h_prev, x_t), axis=1)
        Z = (Whh @ self.Whf) + self.bhf

        h_next = np.tanh(Z)

        return h_next

    def backward(self, h_next, x_t):
        """
        calculates the hidden state in the backward direction for
        one time step
        - x_t is a numpy.ndarray of shape (m, i) that contains
          the data input for the cell
          * m is the batch size for the data
        - h_next is a numpy.ndarray of shape (m, h) containing
          the next hidden state
        Returns: h_pev, the previous hidden state
        """
        Whh = np.concatenate((h_next, x_t), axis=1)
        Z = (Whh @ self.Whb) + self.bhb

        h_next = np.tanh(Z)

        return h_next

    def output(self, H):
        """
        calculates all outputs for the RNN:
        - H is a numpy.ndarray of shape (t, m, 2 * h) that contains
          the concatenated hidden states from both directions,
          excluding their initialized states
          * t is the number of time steps
          * m is the batch size for the data
          * h is the dimensionality of the hidden states
        Returns: Y, the outputs
        """
        Y = (H @ self.Wy) + self.by
        Y = np.exp(Y)

        return Y / Y.sum(axis=-1, keepdims=True)
