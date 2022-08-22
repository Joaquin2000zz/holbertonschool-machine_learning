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

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for i in range(0, self.L):
            if layers[i] < 0 or not isinstance(layers[i], int):
                raise TypeError("layers must be a list of positive integers")
            self.__weights['b{}'.format(i + 1)] = np.zeros(shape=(layers[i],
                                                                  1))
            if i - 1 > - 1:
                he_et_al = np.sqrt(2 / layers[i - 1])
                Wn = np.random.randn(layers[i], layers[i - 1]) * he_et_al
                self.__weights['W{}'.format(i + 1)] = Wn
            else:
                he_et_al = np.sqrt(2 / nx)
                Wn = np.random.randn(layers[i], nx) * he_et_al
                self.__weights['W{}'.format(i + 1)] = Wn

    @property
    def L(self):
        """
        getter method
        """
        return self.__L

    @property
    def cache(self):
        """
        getter method
        """
        return self.__cache

    @property
    def weights(self):
        """
        getter method
        """
        return self.__weights

    def forward_prop(self, X):
        """
        Calculation of forward propagation of DeepNeuralNetwork
        """
        if 'A0' not in self.__cache:
            self.__cache['A0'] = X
        for i in range(1, self.__L + 1):
            if i == 1:
                W = self.__weights.get('W{}'.format(i))
                b = self.__weights.get('b{}'.format(i))
                Zn = W @ X + b
            else:
                W = self.__weights.get('W{}'.format(i))
                X = self.__cache.get('A{}'.format(i - 1))
                Zn = W @ X
                Zn += self.__weights.get('b{}'.format(i))
            self.__cache['A{}'.format(i)] = 1 / (1 + np.exp(-Zn))
        return self.__cache['A{}'.format(i)], self.__cache

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
        cost = self.cost(Y, A.get('A{}'.format(self.__L)))
        return np.where(A.get('A{}'.format(self.__L)) >= 0.5, 1, 0), cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network
        """
        m = Y.shape[1]
        for i in range(self.__L, 0, -1):
            if i != 1:
                print("en iteracion{} , layers {}".format(i, self.__L))
                dZl = cache["A{}".format(i)] - Y
                dWl = (dZl @ cache["A{}".format(i - 1)].T) / m
                dbl = np.sum(dZl, axis=1, keepdims=True) / m
                dZnext = dZl

                Wl = self.__weights['W{}'.format(i)]
                bl = self.__weights['b{}'.format(i)]
                self.__weights['W{}'.format(i)] = Wl - (alpha * dWl)
                self.__weights['b{}'.format(i)] = bl - (alpha * dbl)
            else:
                print("en else iteracion{} , layers {}".format(i, self.__L))
                Wnext = self.__weights['W{}'.format(i + 1)]
                Al = cache['A{}'.format(i)]
                dZl = (Wnext.T @ dZnext) * Al * (1 - Al)
                dWl = (dZl @ self.__cache['A0'].T) / m
                dbl = 1 / m * np.sum(dZl, axis=1, keepdims=True)


                Wl = self.__weights['W{}'.format(i)]
                bl = self.__weights['b{}'.format(i)]
                self.__weights['W{}'.format(i)] = Wl - (alpha * dWl)
                self.__weights['b{}'.format(i)] = bl - (alpha * dbl)