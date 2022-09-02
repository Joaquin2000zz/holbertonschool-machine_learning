#!/usr/bin/env python3
"""
module which contains create_batch_norm_layer function
"""
import tensorflow.compat.v1 as tf


def create_batch_norm_layer(prev, n, activation):
    """
    creates a batch normalization layer for a neural network in tensorflow:

    * prev is the activated output of the previous layer
    * n is the number of nodes in the layer to be created
    * activation is the activation function that should be
      used on the output of the layer
    * you should use the tf.keras.layers.Dense layer as the base
      layer with kernal initializer 
      tf.keras.initializers.VarianceScaling(mode='fan_avg')
    * your layer should incorporate two trainable parameters, gamma and beta,
      initialized as vectors of 1 and 0 respectively
    * you should use an epsilon of 1e-8
    Returns: a tensor of the activated output fo
    """
    #  implement He-et-al initialization for the layer weights
    het_et_al = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    # kernel_initializer=het_et_al
    linear_model = tf.layers.Dense(name="layer",
                                   units=n,
                                   activation=activation,
                                   kernel_initializer=het_et_al)
    layer = linear_model(prev)

    beta = tf.Variable(initial_value=tf.constant(0.0, shape=[n], name='beta'))
    gamma = tf.Variable(initial_value=tf.constant(1.0, shape=[n], name='gamma'))
    mean, std = tf.nn.moments(layer, axes=0, keep_dims=True)
    v1 = tf.nn.batch_normalization(layer, mean, std, offset=beta,
                                   scale=gamma, variance_epsilon=1e-8)
    return activation(v1)
