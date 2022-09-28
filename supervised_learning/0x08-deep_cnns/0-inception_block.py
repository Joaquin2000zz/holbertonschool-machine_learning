#!/usr/bin/env python3
"""
module which contains inception_block function
"""
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """
    builds an inception block as described in
    paper "Going Deeper with Convolutions (2014)":

    * A_prev is the output from the previous layer
    * filters is a tuple or list containing
      F1, F3R, F3,F5R, F5, FPP, respectively:
        - F1 is the number of filters in the 1x1 convolution
        - F3R is the number of filters in the 1x1 convolution
          before the 3x3 convolution
        - F3 is the number of filters in the 3x3 convolution
        - F5R is the number of filters in the 1x1 convolution
          before the 5x5 convolution
        - F5 is the number of filters in the 5x5 convolution
        - FPP is the number of filters in the 1x1 convolution after
          the max pooling (Note : The output shape after the max pooling layer
          is outputshape = math.floor((inputshape - 1) / strides) + 1)
    * All convolutions inside the inception block should use
      a rectified linear activation (ReLU)
    Returns: the concatenated output of the inception block
    """

    layer_0 = K.layers.Conv2D(
        filters[0], (1, 1), padding='same', activation='relu')(A_prev)

    layer_1 = K.layers.Conv2D(
        filters[1], (1, 1), padding='same', activation='relu')(A_prev)
    layer_1 = K.layers.Conv2D(
        filters[2], (3, 3), padding='same', activation='relu')(layer_1)

    layer_2 = K.layers.Conv2D(
        filters[3], (1, 1), padding='same', activation='relu')(A_prev)
    layer_2 = K.layers.Conv2D(
        filters[4], (5, 5), padding='same', activation='relu')(layer_2)

    layer_3 = K.layers.MaxPooling2D(
        (3, 3), strides=(1, 1), padding='same')(A_prev)
    layer_3 = K.layers.Conv2D(
        filters[5], (1, 1), padding='same', activation='relu')(layer_3)

    block = K.layers.concatenate([layer_0, layer_1, layer_2, layer_3], axis=3)

    return block
