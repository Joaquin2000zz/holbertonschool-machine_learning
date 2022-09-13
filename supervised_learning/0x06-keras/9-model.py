#!/usr/bin/env python3
"""
module which contains save_model and load_model function
"""
import tensorflow.keras as K


def save_model(network, filename): 
    """
    saves an entire model:

    * network is the model to save
    * filename is the path of the file that the model should be saved to
    Returns: None
    """

    filename += ".h3" if not ".h5" in filename else ''

    network.save(filename)

def load_model(filename):
    """
    loads an entire model:

    * filename is the path of the file that the model should be loaded from
    Returns: the loaded model
    """

    filename += ".h3" if not ".h5" in filename else ''

    return K.models.load_model(filename)
