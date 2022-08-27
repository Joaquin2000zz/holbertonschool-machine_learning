#!/usr/bin/env python3
"""
module which contains DeepNeuralNetwork class
"""
import numpy as np
import matplotlib.pyplot as plt


class DeepNeuralNetwork:
    """
    defines a deep neural network performing multiclass classificaiton
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
            if self.__L - 1 == i:
                e = np.exp(Zn)
                self.__cache['A{}'.format(i)] = e / np.sum(e, axis=0,
                                                           keepdims=True)
            else:
                self.__cache['A{}'.format(i)] = 1 / (1 + np.exp(-Zn))
        return self.__cache['A{}'.format(self.__L)], self.__cache

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        """
        m = - (1 / Y.shape[1])
        Hto = np.sum(Y * np.log(A))
        return m * Hto

    def evaluate(self, X, Y):
        """
        Evaluates neuron’s predictions performing binary classification
        """
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        return np.where(A == np.amax(A, axis=0), 1, 0), cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network
        """
        m = Y.shape[1]
        dZ = cache["A{}".format(self.L)] - Y

        for i in range(self.L, 0, -1):
            # this is because you need to use the dZ of the prev iteration
            db = np.sum(dZ, axis=1, keepdims=True) / m

            A = self.cache["A{}".format(i - 1)]
            dW = dZ @ A.T / m

            # preparing dZ to the next iteration
            dZ = (self.weights["W{}".format(i)].T @ dZ) * (A * (1 - A))

            self.__weights["W{}".format(i)] -= dW * alpha
            self.__weights["b{}".format(i)] -= db * alpha

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        '''
        Trains the deep neural network
        '''
        if type(iterations) is not int:
            raise TypeError('iterations must be an integer')
        if iterations < 0:
            raise ValueError('iterations must be a positive integer')
        if type(alpha) is not float:
            raise TypeError('alpha must be a float')
        if alpha < 0:
            raise ValueError('alpha must be positive')
        if verbose is True or graph is True:
            if not isinstance(step, int):
                raise TypeError('step must be an integer')
            if step < 0 or step > iterations:
                raise ValueError('step must be positive and <= iterations')
        iteration = []
        costlist = []
        for i in range(iterations + 1):
            _, cost = self.evaluate(X, Y)
            self.forward_prop(X)

            if i < iterations:
                self.gradient_descent(Y, self.__cache, alpha)
            if i <= step:
                iteration.append(i)
                costlist.append(cost)
                if verbose and (i == 0 or i % step == 0):
                    print('Cost after {} iterations: {}'.format(i, cost))
        if graph:
            plt.plot(iteration, costlist, 'b')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()
        return self.evaluate(X, Y)

    def save(self, filename):
        """
        Saves the instance object to a file in pickle format
        """
        import pickle
        if filename[-4:] != '.pkl':
            filename += '.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """
        Loads a pickled DeepNeuralNetwork object
        """
        import pickle
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None
