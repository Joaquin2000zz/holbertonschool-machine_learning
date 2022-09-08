#!/usr/bin/env python3
"""
module which contains l2_reg_create_layer function
"""
import tensorflow.compat.v1 as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a tensorflow layer that includes L2 regularization:

    * prev is a tensor containing the output of the previous layer
    * n is the number of nodes the new layer should contain
    * activation is the activation function that should be used on the layer
    * lambtha is the L2 regularization parameter
    Returns: the output of the new layer
    """
    het_et_al = tf.keras.initializers.VarianceScaling(scale=2.0,
                                                      mode=("fan_avg"))
    L2 = tf.keras.regularizers.L2(lambtha)
    layer = tf.layers.Dense(n,
                            activation=activation,
                            kernel_initializer=het_et_al,
                            kernel_regularizer=L2)
    return layer(prev)
