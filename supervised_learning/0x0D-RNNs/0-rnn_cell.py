#!/usr/bin/env python3
"""
module which contains RNNCell function
"""
import numpy as np


class RNNCell:
    """
    represents a cell of a simple RNN
    """

    def __init__(self, i, h, o):
        """
        class constructor
        - i is the dimensionality of the data
        - h is the dimensionality of the hidden state
        - o is the dimensionality of the outputs
        - Creates the public instance attributes Wh, Wy, bh, by
          that represent the weights and biases of the cell
          * Wh and bh are for the concatenated hidden state and input data
          * Wy and by are for the output
        - The weights should be initialized using a random normal distribution
          in the order listed above
        - The weights will be used on the right side for matrix multiplication
        - The biases should be initialized as zeros
        """
        self.Wh = np.random.normal(size=(i + h, h))
        self.bh = np.zeros(shape=(1, h))
        self.Wy = np.random.normal(size=(h, o))
        self.by = np.zeros(shape=(1, o))

    def forward(self, h_prev, x_t):
        """
        performs forward propagation for one time step
        - x_t is a numpy.ndarray of shape (m, i) that contains
          the data input for the cell
          * m is the batch size for the data
        - h_prev is a numpy.ndarray of shape (m, h) containing
          the previous hidden state
        - The output of the cell should use a softmax activation function
        Returns: h_next, y
          - h_next is the next hidden state
          - y is the output of the cell
        """
        Whh = np.concatenate((h_prev, x_t), axis=1)
        Z = (Whh @ self.Wh) + self.bh

        h_next = np.tanh(Z)
        yt = (h_next @ self.Wy) + self.by
        yexp = np.exp(yt)

        return h_next, yexp / yexp.sum(axis=1, keepdims=True)
