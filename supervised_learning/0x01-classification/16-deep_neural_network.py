#!/usr/bin/env python3
"""
module which contains DeepNeuralNetwork class
"""
import numpy as np


class DeepNeuralNetwork:
    """
    defines a deep neural network performing binary classificaiton
    He-et-al Initialization to the weights
    """
    def __init__(self, nx, layers):
        """
        constructor method
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or not layers:
            raise TypeError("layers must be a list of positive integers")
        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        for i in range(1, self.L):
            if layers[i - 1] < 0:
                raise TypeError("layers must be a positive integer")
            Wn = np.zeros(shape=(layers[i - 1], 1)) * np.sqrt(2 / layers[i - 1])
            self.weights[f'b{i}'] = Wn
            self.weights[f'W{i}'] = np.random.randn(layers[i - 1], nx)
