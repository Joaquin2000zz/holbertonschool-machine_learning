#!/usr/bin/env python3
"""
module which contains NeuralNetwork
"""
import numpy as np


class NeuralNetwork:
    """
    defines a neural network with one hidden layer
    performing binary classification
    """
    def __init__(self, nx, nodes):
        """
        NeuralNetwork constructor
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros(shape=(nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """
        getter method
        """
        return self.__W1

    @property
    def b1(self):
        """
        getter method
        """
        return self.__b1

    @property
    def A1(self):
        """
        getter method
        """
        return self.__A1

    @property
    def W2(self):
        """
        getter method
        """
        return self.__W2

    @property
    def b2(self):
        """
        getter method
        """
        return self.__b2

    @property
    def A2(self):
        """
        getter method
        """
        return self.__A2

    def forward_prop(self, X):
        """
        forward propagation of the neural network
        """
        Z1 = self.__W1 @ X + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z1))

        Z2 = self.__W2 @ self.__A1 + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z2))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        """
        a = (Y * np.log(A))
        b = ((1 - Y) * np.log(1.0000001 - (A)))
        sigma = a + b
        return (-1 / Y.shape[1]) * np.sum(sigma)

    def evaluate(self, X, Y):
        """
        Evaluates neuron’s predictions performing binary classification
        """
        _, A = self.forward_prop(X)
        return np.where(A >= 0.5, 1, 0), self.cost(Y, A)

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network
        """
        m = Y.shape[1]

        dZ2 = A2 - Y
        dW2 = (dZ2 @ A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m

        dZ1 = (self.__W2.T @ dZ2) * A1 * (1 - A1)
        dW1 = (dZ1 @ X.T) / m
        db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

        self.__W1 = self.__W1 - (alpha * dW1)
        self.__b1 = self.__b1 - (alpha * db1)

        self.__W2 = self.__W2 - (alpha * dW2)
        self.__b2 = self.__b2 - (alpha * db2)

def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
    """
    Trains the neural networx performing binary classification
    """
    if not isinstance(iterations, int):
        raise TypeError('iterations must be an integer')
    if iterations < 0:
        raise ValueError('iterations must be a positive integer')
    if not isinstance(alpha, float):
        raise TypeError('alpha must be a float')
    if alpha < 0:
        raise ValueError('alpha must be positive')
    if (verbose or graph) and not isinstance(step, int):
        raise TypeError('step must be an integer')
    if (verbose or graph) and (step < 0 or step > iterations):
        raise ValueError('step must be positive and <= iterations')


    for _ in range(iterations):
        self.forward_prop(X)
        self.gradient_descent(X, Y, self.__A1, self.__A2, alpha)

    return self.evaluate(X, Y)
    