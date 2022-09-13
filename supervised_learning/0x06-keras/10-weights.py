#!/usr/bin/env python3
"""
module which contains save_model and load_model function
"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """
    saves a model's weights:

    * network is the model whose weights should be saved
    * filename is the path of the file that the weights should be saved to
    * save_format is the format in which the weights should be saved
    Returns: None
    """

    filename += save_format if save_format not in filename else ''

    network.save_weights(filename)


def load_weights(network, filename):
    """
    Loads a model’s weights:
    network is the model to which the weights should be loaded
    filename is the path of the file that the weights should be loaded from
    Returns: None
    """

    return network.load_weights(filename)
