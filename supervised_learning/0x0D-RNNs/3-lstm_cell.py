#!/usr/bin/env python3
"""
module which contains LSTMCell class
"""
import numpy as np


class LSTMCell:
    """
    represents an long-short term memory unit
    """

    def __init__(self, i, h, o):
        """
        class constructor
        - i is the dimensionality of the data
        - h is the dimensionality of the hidden state
        - o is the dimensionality of the outputs
        - Creates the public instance attributes Wf, Wu, Wc, Wo, Wy, bf, bu,
          bc, bo, by that represent the weights and biases of the cell
          * Wf and bf are for the forget gate
          * Wu and bu are for the update gate
          * Wc and bc are for the intermediate cell state
          * Wo and bo are for the output gate
          * Wy and by are for the outputs
        - The weights should be initialized using a random normal distribution
          in the order listed above
        - The weights will be used on the right side for matrix multiplication
        - The biases should be initialized as zeros
        """
        self.Wf = np.random.normal(size=(i + h, h))
        self.bf = np.zeros(shape=(1, i))
        self.Wu = np.random.normal(size=(i + h, h))
        self.bu = np.zeros(shape=(1, i))
        self.Wc = np.random.normal(size=(i + h, h))
        self.bc = np.zeros(shape=(1, i))
        self.Wo = np.random.normal(size=(i + h, h))
        self.bo = np.zeros(shape=(1, i))
        self.Wy = np.random.normal(size=(h, o))
        self.by = np.zeros(shape=(1, o))

    @staticmethod
    def sigmoid(Z):
        """
        performs sigmoid activation
        """
        return 1 / (1 + np.exp(-Z))

    def forward(self, h_prev, c_prev, x_t):
        """
        performs forward propagation for one time step
        - x_t is a numpy.ndarray of shape (m, i) that contains
          the data input for the cell
          * m is the batche size for the data
        - h_prev is a numpy.ndarray of shape (m, h) containing
          the previous hidden state
        - c_prev is a numpy.ndarray of shape (m, h) containing
          the previous cell state
        - The output of the cell should use a softmax activation function
        Returns: h_next, c_next, y
        - h_next is the next hidden state
        - c_next is the next cell state
        - y is the output of the cell
        """
        Whh = np.concatenate((h_prev, x_t), axis=-1)

        ft = self.sigmoid((Whh @ self.Wf) + self.bf)
        it = self.sigmoid((Whh @ self.Wu) + self.bu)
        ot = self.sigmoid((Whh @ self.Wo) + self.bo)

        CHat = np.tanh((Whh @ self.Wc) + self.bc)
        c_next = (ft * c_prev) + (it * CHat)

        h_next = ot * np.tanh(c_next)

        yt = (h_next @ self.Wy) + self.by
        yexp = np.exp(yt)

        return h_next, c_next, yexp / yexp.sum(axis=1, keepdims=True)
