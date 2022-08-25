#!/usr/bin/env python3
"""
module which contains calculate_accuracy
"""
import tensorflow.compat.v1 as tf


def calculate_accuracy(y, y_pred):
    """
    that calculates the accuracy of a prediction:

    y is a placeholder for the labels of the input data
    y_pred is a tensor containing the network’s predictions
    Returns: a tensor containing the decimal accuracy of the prediction
    """
    return tf.reduce_mean(y_pred / y)