#!/usr/bin/env python3
"""
module which contains create_momentum_op function
"""
import numpy as np
import tensorflow.compat.v1 as tf


def create_momentum_op(loss, alpha, beta1):
    """
    creates the training operation for a neural network in tensorflow
    using the gradient descent with momentum optimization algorithm:
    * loss is the loss of the network
    * alpha is the learning rate
    * beta1 is the momentum weight
    Returns: the momentum optimization operation
    """
    momentum = tf.compat.v1.train.MomentumOptimizer(alpha, beta1)
    return momentum.compute_gradients(loss)
