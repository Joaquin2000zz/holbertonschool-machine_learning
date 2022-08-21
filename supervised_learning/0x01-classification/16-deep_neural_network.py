#!/usr/bin/env python3
"""
module which contains DeepNeuralNetwork class
"""


import numpy as np

class DeepNeuralNetwork:
    """
    defines a deep neural network performing binary classificaiton
    """
    def __init__(self, nx, layers):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not layers or not isinstance(layers, list):
            raise TypeError("layers must be a list of positive integers")
        if any(layer < 0 for layer in layers):
            raise ValueError("layers must be a positive integer")
        self.L = len(layers)
        self.cache = {}
        self.weights = self.start(nx , layers, self.L)

    def start(self, nx, layers, i):
        """
        initializes wheights and biases of DNN
        """
        if not i:
            return
        self.cache[f'b{i}'] = np.zeros(shape=(layers[i - 1], 1))
        self.cache[f'W{i}'] = np.random.randn(layers[i - 1], nx)
        self.start(nx, layers, i - 1)
