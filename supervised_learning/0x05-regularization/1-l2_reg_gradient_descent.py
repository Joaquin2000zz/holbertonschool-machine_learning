#!/usr/bin/env python3
"""
module which contains l2_reg_gradient_descent function
"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    updates the weights and biases of a neural network using
    gradient descent with L2 regularization:

    * Y is a one-hot numpy.ndarray of shape (classes, m) that
      contains the correct labels for the data
    * classes is the number of classes
    * m is the number of data points
    * weights is a dictionary of the weights and biases
      of the neural network
    * cache is a dictionary of the outputs
      of each layer of the neural network
    * alpha is the learning rate
    * lambtha is the L2 regularization parameter
    * L is the number of layers of the network
    * The neural network uses tanh activations on each layer
      except the last, which uses a softmax activation
    * The weights and biases of the network should be updated in place
    """

    m = Y.shape[1]
    for i in reversed(range(L)):
        w = 'W' + str(i + 1)
        b = 'b' + str(i + 1)
        a = 'A' + str(i + 1)
        a_0 = 'A' + str(i)
        A = cache[a]
        A_dw = cache[a_0]
        if i == L - 1:
            dz = A - Y
            W = weights[w]
        else:
            da = 1 - (A * A)
            dz = np.matmul(W.T, dz)
            dz = dz * da
            W = weights[w]
        dw = np.matmul(A_dw, dz.T) / m
        db = np.sum(dz, axis=1, keepdims=True) / m
        weights[w] = weights[w] - alpha * (dw.T + (lambtha / m * weights[w]))
        weights[b] = weights[b] - alpha * db
