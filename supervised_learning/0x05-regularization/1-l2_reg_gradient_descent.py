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
    dZ = cache["A{}".format(L)] - Y

    for i in range(L, 0, -1):
        # this is because you need to use the dZ of the prev iteration
        db = np.sum(dZ, axis=1, keepdims=True) / m

        Aprev = i - 1
        A = cache["A{}".format(Aprev)]
        dW = dZ @ A.T / m

        # preparing dZ to the next iteration
        dx = i
        dZ = (weights["W{}".format(dx)].T @ dZ) * (A * (1 - A))
        L2 = (1 - (alpha * lambtha) / m)
        weights["W{}".format(dx)] *= L2 - (dW * alpha)
        weights["b{}".format(dx)] -= db * alpha
