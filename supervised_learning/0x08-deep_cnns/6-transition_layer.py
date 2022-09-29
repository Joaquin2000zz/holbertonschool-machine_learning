#!/usr/bin/env python3
"""
module which containts dense_block function
"""
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """
    builds a transition layer as described in the paper
    "Densely Connected Convolutional Networks":

    * X is the output from the previous layer
    * nb_filters is an integer representing the number of filters in X
    * compression is the compression factor for the transition layer
    * Your code should implement compression as used in DenseNet-C
    * All weights should use he normal initialization
    * All convolutions should be preceded by Batch Normalization and
      a rectified linear activation (ReLU), respectively
    Returns: The output of the transition layer and
             the number of filters within the output, respectively
    """
    het_et_al = K.initializers.HeNormal()

    X = K.layers.BatchNormalization()(X)
    X = K.layers.Activation('relu')(X)
    X = K.layers.Conv2D(nb_filters * compression, kernel_size=(1, 1),
                        kernel_initializer=het_et_al,
                        strides=(1, 1), padding='same')(X)
    X = K.layers.AvgPool2D(pool_size=(2, 2), strides=(2, 2),
                           padding="valid")(X)

    return X, nb_filters * compression
