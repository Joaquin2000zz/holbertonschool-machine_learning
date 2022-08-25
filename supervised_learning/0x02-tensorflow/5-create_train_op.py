#!/usr/bin/env python3
"""
module which contains calculate_loss
"""
import tensorflow.compat.v1 as tf


def create_train_op(loss, alpha):
    """
    creates the training operation for the network:

    loss is the loss of the network’s prediction
    alpha is the learning rate
    Returns: an operation that trains the network using gradient descent
    """
    optimizer = tf.train.GradientDescentOptimizer(alpha)
    return optimizer.minimize(loss)
