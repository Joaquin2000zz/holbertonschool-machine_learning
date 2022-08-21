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
        if not isinstance(layers, list):
            raise TypeError("layers must be a list of positive integers")
        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        for i in range(0, self.L):
            if layers[i - 1] < 1:
                raise TypeError("layers must be a list of positive integers")
            self.weights[f'b{i + 1}'] = np.zeros(shape=(layers[i], 1))
            if i - 1 > -1:
                he_et_al = np.sqrt(2 / layers[i - 1])
                Wn = np.random.randn(layers[i], layers[i - 1]) * he_et_al
                self.weights[f'W{i + 1}'] = Wn
            else:
                he_et_al = np.sqrt(2 / nx)
                Wn = np.random.randn(layers[i], nx) * he_et_al
                self.weights[f'W{i + 1}'] = Wn
