#!/usr/bin/env python3
"""
module which contains create_RMSProp_op function
"""
import tensorflow.compat.v1 as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """
    creates the training operation for a neural network in tensorflow
    using the RMSProp optimization algorithm:

    * loss is the loss of the network
    * alpha is the learning rate
    * beta2 is the RMSProp weight
    * epsilon is a small number to avoid division by zero
    Returns: the RMSProp optimization operation
    """
    RMS = tf.train.RMSPropOptimizer(learning_rate=alpha,
                                    decay=beta2,
                                    epsilon=epsilon)
    return RMS.minimize(loss)
