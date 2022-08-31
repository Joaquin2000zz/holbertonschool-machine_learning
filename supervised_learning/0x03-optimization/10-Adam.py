#!/usr/bin/env python3
"""
module which contains create_Adam_op function
"""
import tensorflow.compat.v1 as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
    creates the training operation for a neural network in tensorflow
    using the Adam optimization algorithm:

    * loss is the loss of the network
    * alpha is the learning rate
    * beta1 is the weight used for the first moment
    * beta2 is the weight used for the second moment
    * epsilon is a small number to avoid division by zero
    Returns: the Adam optimization operation
    """
    Adam = tf.train.RMSPropOptimizer(learning_rate=alpha,
                                     decay=beta1, momentum=beta2,
                                     epsilon=epsilon)
    return Adam.minimize(loss)
