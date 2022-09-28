#!/usr/bin/env python3
"""
module which contains inception_network function
"""
import tensorflow.keras as K


def identity_block(A_prev, filters):
    """
    that builds an identity block as described in the paper
    "Deep Residual Learning for Image Recognition (2015)":

    * A_prev is the output from the previous layer
    * filters is a tuple or list containing F11, F3, F12, respectively:
        - F11 is the number of filters in the first 1x1 convolution
        - F3 is the number of filters in the 3x3 convolution
        - F12 is the number of filters in the second 1x1 convolution
    * All convolutions inside the block should be followed by batch
      normalization along the channels axis and a rectified
      linear activation (ReLU), respectively.
    * All weights should use he normal initialization
    Returns: the activated output of the identity block
    """
    het_et_al = K.initializers.HeNormal()
    # Retrieve Filters
    F11, F3, F12 = filters
    
    # Save the input value.
    # needed this later to add back to the main path. 
    identity = X = A_prev
    
    # First component of main path
    X = K.layers.Conv2D(filters=F11, kernel_size=(1, 1),
                        strides=(1, 1), kernel_initializer=het_et_al,
                        padding='valid')(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)
    
    
    # Second component of main path
    X = K.layers.Conv2D(filters=F3, kernel_size=(3, 3),
                        strides=(1, 1), kernel_initializer=het_et_al,
                        padding='same')(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # Third component of main path 
    X = K.layers.Conv2D(filters=F12, kernel_size=(1, 1),
                        strides=(1, 1), kernel_initializer=het_et_al,
                        padding='valid')(X)
    X = K.layers.BatchNormalization(axis=3)(X)

    # Final step: Add shortcut value to main path,
    #             and pass it through a RELU activation 
    X = K.layers.Add()([X, identity])
    X = K.layers.Activation('relu')(X)

    return X
