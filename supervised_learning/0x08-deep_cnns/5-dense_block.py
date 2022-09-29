#!/usr/bin/env python3
"""
module which containts dense_block function
"""
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    builds a dense block as described in Densely Connected Convolutional Networks:

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

    def composite_function(x, filters, kernel_size=(3, 3), strides=(1, 1)):
        x = K.layers.BatchNormalization()(x)
        x = K.layers.ReLU()(x)
        x = K.layers.Conv2D(filters, kernel_size=kernel_size, kernel_initializer=het_et_al,
                            strides=strides, padding='same')(x)
        return x

    for _ in range(layers):
        y = composite_function(X, growth_rate * nb_filters)
        y = composite_function(y, nb_filters)
        X = K.layers.concatenate([y, X], axis=-1)

    return X
