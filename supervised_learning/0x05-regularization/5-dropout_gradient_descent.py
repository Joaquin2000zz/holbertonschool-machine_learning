#!/usr/bin/env python3
"""
module which contains dropout_gradient_descent function
"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    updates the weights of a neural network with Dropout regularization
    using gradient descent:

    * Y is a one-hot numpy.ndarray of shape (classes, m)
      that contains the correct labels for the data
        - classes is the number of classes
        - m is the number of data points
    * weights is a dictionary of the weights and biases of the neural network
    * cache is a dictionary of the outputs and dropout masks of each layer of
      the neural network
    * alpha is the learning rate
    * keep_prob is the probability that a node will be kept
    * L is the number of layers of the network
    * All layers use thetanh activation function except the last,
      which uses the softmax activation function
    The weights of the network should be updated in place
    """

    m = Y.shape[1]
    for i in reversed(range(1, L + 1)):
        w = 'W' + str(i)
        b = 'b' + str(i)
        a = 'A' + str(i)
        a_0 = 'A' + str(i - 1)
        A = cache[a]
        A_dw = cache[a_0]
        if i == L:
            dz = A - Y
            W = weights[w]
        else:
            D = cache["D{}".format(i)]
            # implementing dropout
            da = 1 - (A * A)
            dz = np.matmul(W.T, dz)
            dz = dz * da
            dz *= D
            dz /= keep_prob
            W = weights[w]
        dw = np.matmul(A_dw, dz.T) / m
        db = np.sum(dz, axis=1, keepdims=True) / m
        weights[w] = weights[w] - alpha * dw.T
        weights[b] = weights[b] - alpha * db
