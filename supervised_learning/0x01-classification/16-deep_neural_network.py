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
        self.weights = {}
        for i in range(1, self.L):
            self.weights[f'b{i - 1}'] = np.zeros(shape=(layers[i - 1], 1))
            self.weights[f'W{i - 1}'] = np.random.randn(layers[i - 1], nx)
