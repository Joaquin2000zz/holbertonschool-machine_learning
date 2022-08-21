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
        for i in range(1, self.L + 1):
            if layers[i - 1] < 0:
                raise TypeError("layers must be a positive integer")
            self.weights[f'b{i}'] = np.zeros(shape=(layers[i - 1], 1))
            if i - 2 > -1:
                he_et_al = np.sqrt(2 / layers[i - 2])
                Wn = np.random.randn(layers[i - 1], layers[i - 2]) * he_et_al
                self.weights[f'W{i}'] = Wn
            else:
                he_et_al = np.sqrt(2 / nx)
                Wn = np.random.randn(layers[i - 1], nx) * he_et_al
                self.weights[f'W{i}'] = Wn
