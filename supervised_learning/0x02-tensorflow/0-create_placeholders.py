#!/usr/bin/env python3
"""
module which contains create_placeholders function
"""
import tensorflow.compat.v1 as tf


def create_placeholders(nx, classes):
    """
    returns two placeholders, x and y, for the neural network
    """
    return tf.placeholder(tf.float32, shape=(None, nx)), tf.placeholder(tf.float32, shape=(None, classes))