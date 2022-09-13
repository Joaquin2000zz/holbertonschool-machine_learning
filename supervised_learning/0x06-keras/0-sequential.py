#!/usr/bin/env python3
"""
module which contains build_model function
"""
import tensorflow.keras as K
import tensorflow as tf


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    builds a neural network with the Keras library:

    * nx is the number of input features to the network
    * layers is a list containing the number of nodes in each
      layer of the network
    * activations is a list containing the activation functions
      used for each layer of the network
    * lambtha is the L2 regularization parameter
    * keep_prob is the probability that a node will be kept for dropout
    * You are not allowed to use the Input class
    Returns: the keras model
    """
    L = []

    # l2 regularization
    l2 = K.regularizers.L2(lambtha)

    for n, activation in zip(layers, activations):

        # creating layer
        l = K.layers.Dense(n, activation=activation,
                           activity_regularizer=l2)
        L.append(l)

        # creating dropout to that layer 1 - p
        dropout = K.layers.Dropout(rate=1 - keep_prob, input_shape=(None, n))
        L.append(dropout)

    # removing dropout to the output layer
    L.pop()

    # creating model with created layers in loop
    model = K.Sequential(L)

    # initializes values
    model(tf.ones((1, nx)))

    return model
