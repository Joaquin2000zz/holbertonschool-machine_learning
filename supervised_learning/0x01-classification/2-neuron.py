#!/usr/bin/env python3
"""
module which contains neuron class
"""
import numpy as np
from math import e

class Neuron:
    """
    defines a single neuron performing binary classification
    """
    def __init__(self, nx):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def get_W(self):
        """
        getter method
        """
        return self.__W

    @property
    def get_b(self):
        """
        getter method
        """
        return self.__b

    @property
    def A(self):
        """
        getter method
        """
        return self.__A

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron
        using a sigmoid function as trigger
        """
        Z = np.dot(self.__W, X)
        self.__A = 1 / (1 + e ** Z)
        return self.__A
