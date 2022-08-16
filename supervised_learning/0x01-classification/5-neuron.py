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
        Z = np.dot(self.__W, X) + self.__b
        self.__A = 1 / (1 + e ** Z)
        return self.__A

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        """
        a = (Y @ np.log(A).transpose())
        b = ((1.0000001 - Y) @ np.log(1.0000001 - A).transpose())
        sigma = a + b
        return (- 1 / Y.shape[1]) * np.sum(sigma)

    def evaluate(self, X, Y):
        """
        Evaluates the neuron’s predictions performing binary classification
        """
        Z = np.dot(self.__W, X) + self.__b
        return np.where(Z >= 0.5, 1, 0), self.cost(Y, self.forward_prop(X))

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neuron
        performing binary classification
        """
        self.__W = self.__W - alpha * (A - Y) @ X.transpose()
        self.__b = self.__b - (alpha / X.shape[1]) * np.sum((A - Y))
