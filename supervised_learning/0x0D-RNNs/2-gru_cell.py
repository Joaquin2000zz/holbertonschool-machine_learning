#!/usr/bin/env python3
"""
module which contains GRUCell class
"""
import numpy as np


class GRUCell:
    """
    represents a gated recurrent unit
    """

    def __init__(self, i, h, o):
        """
        class constructor
        - i is the dimensionality of the data
        - h is the dimensionality of the hidden state
        - o is the dimensionality of the outputs
        - Creates the public instance attributes Wz, Wr, Wh, Wy, bz,
          br, bh, by that represent the weights and biases of the cell
          * Wz and bz are for the update gate
          * Wr and br are for the reset gate
          * Wh and bh are for the intermediate hidden state
          * Wy and by are for the output
        - The weights should be initialized using a random
        normal distribution in the order listed above
        - The weights will be used on the right side for matrix multiplication
        - The biases should be initialized as zeros
        """
        self.Wz = np.random.normal(size=(i + h, h))
        self.bz = np.zeros(shape=(1, i))
        self.Wr = np.random.normal(size=(i + h, h))
        self.br = np.zeros(shape=(1, i))
        self.Wh = np.random.normal(size=(i + h, h))
        self.bh = np.zeros(shape=(1, i))
        self.Wy = np.random.normal(size=(h, o))
        self.by = np.zeros(shape=(1, o))

    @staticmethod
    def sigmoid(Z):
        """
        performs sigmoid activation
        """
        return 1 / (1 + np.exp(-Z))

    def forward(self, h_prev, x_t):
        """
        performs forward propagation for one time step
        - x_t is a numpy.ndarray of shape (m, i) that contains
          the data input for the cell
          * m is the batch size for the data
        - h_prev is a numpy.ndarray of shape (m, h)
          containing the previous hidden state
        - The output of the cell should use a softmax activation function
        Returns: h_next, y
        - h_next is the next hidden state
        - y is the output of the cell
        """

        Whh = np.concatenate((h_prev, x_t), axis=1)

        zt = Whh @ self.Wz
        zt = self.sigmoid(zt + self.bz)

        rt = Whh @ self.Wr
        rt = self.sigmoid(rt + self.br)

        htHat = np.concatenate((rt * h_prev, x_t), axis=1) @ self.Wh
        h_t = np.tanh(htHat + self.bh)

        h_next = (1 - zt) * h_prev + (zt * h_t)

        yt = (h_next @ self.Wy) + self.by
        ytexp = np.exp(yt)

        return h_next, ytexp / ytexp.sum(axis=1, keepdims=True)
