#!/usr/bin/env python3
"""
module which contains create_placeholders function
"""
import tensorflow.compat.v1 as tf


def create_placeholders(nx, classes):
    """
    returns two placeholders, x and y, for the neural network
    nx: the number of feature columns in our data
    classes: the number of classes in our classifier
    Returns: placeholders named x and y, respectively
             x is the placeholder for the input data to the neural network
             y is the placeholder for the one-hot labels for the input data
    """
    nx = tf.placeholder(tf.float32, shape=(None, nx), name='x')
    classes = tf.placeholder(tf.float32, shape=(None, classes), name='y'))

    return nx, classes
