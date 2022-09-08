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
    dZ = cache["A{}".format(L)] - Y
    for i in range(L, 0, -1):
        # this is because you need to use the dZ of the prev iteration
        db = np.sum(dZ, axis=1, keepdims=True) / m
        A = cache["A{}".format(i - 1)]

        dW = dZ @ A.T / m
        # preparing dZ to the next iteration

        dZ = (weights["W{}".format(i)].T @ dZ) * (A * (1 - A))
        if i - 1 != 0:
            D = cache["D{}".format(i - 1)]
            # implementing dropout
            dZ *= D
            dZ /= keep_prob

        weights["W{}".format(i)] -= dW * alpha
        weights["b{}".format(i)] -= db * alpha
