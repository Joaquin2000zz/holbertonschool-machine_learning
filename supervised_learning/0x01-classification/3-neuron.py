#!/usr/bin/env python3
"""
module which contains neuron class
"""
import numpy as np


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
    def W(self):
        """
        getter method
        """
        return self.__W

    @property
    def b(self):
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
        Z = self.__W @ X + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        """
        a = (Y @ np.log(A).transpose())
        b = ((1 - Y) @ np.log(1 - A).transpose())
        sigma = a + b
        return (- 1 / Y.shape[1]) * np.sum(sigma)
