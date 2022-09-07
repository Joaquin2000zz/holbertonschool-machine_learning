#!/usr/bin/env python3
"""
module which contains dropout_forward_prop function
"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    conducts forward propagation using Dropout:

    * X is a numpy.ndarray of shape (nx, m) containing
      the input data for the network
        - nx is the number of input features
        - m is the number of data points
    * weights is a dictionary of the weights and biases of the neural network
    * L the number of layers in the network
    * keep_prob is the probability that a node will be kept
    * All layers except the last should use the tanh activation function
    * The last layer should use the softmax activation function
    Returns: a dictionary containing the outputs of each layer and the dropout mask used on each layer (see example for format)
    """
    cache = {}
    cache['A0'] = X
    
    for i in range(1, L + 1):
        if i == 1:
            W = weights.get('W{}'.format(i))
            b = weights.get('b{}'.format(i))
            Zn = W @ X + b
        else:
            key = 'A{}'.format(i - 1)
            X = cache.get(key)
            D = np.random.rand(X.shape[0], X.shape[1])
            cache[key] *= D
            cache[key] /= keep_prob
            cache['D{}'.format(i)] = D
        
            W = weights.get('W{}'.format(i))
            Zn = W @ X
            Zn += weights.get('b{}'.format(i))
        if L - 1 == i:
            e = np.exp(Zn)
            cache['A{}'.format(i)] = e / np.sum(e, axis=0,
                                                keepdims=True)
        else:
            cache['A{}'.format(i)] = 1 / (1 + np.exp(-Zn))
    return cache
