#!/usr/bin/env python3
"""
module which contains dropout_create_layer function
"""
import tensorflow.compat.v1 as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """
    creates a layer of a neural network using dropout:

    * prev is a tensor containing the output of the previous layer
    * n is the number of nodes the new layer should contain
    * activation is the activation function that should be used on the layer
    * keep_prob is the probability that a node will be kept
    Returns: the output of the new layer
    """
    het_et_al = tf.keras.initializers.VarianceScaling(scale=2.0,
                                                      mode=("fan_avg"))
    layer = tf.layers.Dense(n,
                            activation=activation,
                            kernel_initializer=het_et_al)

    return tf.layers.Dropout(rate=keep_prob)(layer(prev))
