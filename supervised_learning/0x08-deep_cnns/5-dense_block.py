#!/usr/bin/env python3
"""
module which containts dense_block function
"""
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    builds a dense block as described in the paper
    "Densely Connected Convolutional Networks":

    * X is the output from the previous layer
    * nb_filters is an integer representing the number of filters in X
    * growth_rate is the growth rate for the dense block
    * layers is the number of layers in the dense block
    * You should use the bottleneck layers used for DenseNet-B
    * All weights should use he normal initialization
    * All convolutions should be preceded by Batch Normalization and a
      rectified linear activation (ReLU), respectively
    Returns: The concatenated output of each layer within the Dense Block
             and the number of filters within the concatenated outputs,
             respectively
    """
    het_et_al = K.initializers.HeNormal()

    for _ in range(layers):
        y = K.layers.BatchNormalization(axis=3)(X)
        y = K.layers.Activation('relu')(y)
        y = K.layers.Conv2D(growth_rate * 4,
                            kernel_size=(1, 1),
                            strides=(1, 1),
                            padding='same',
                            kernel_initializer=het_et_al)(y)

        y = K.layers.BatchNormalization(axis=3)(y)
        y = K.layers.Activation('relu')(y)
        y = K.layers.Conv2D(growth_rate,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='same',
                            kernel_initializer=het_et_al)(y)

        X = K.layers.concatenate([X, y], axis=3)
        nb_filters += growth_rate
    return X, nb_filters
