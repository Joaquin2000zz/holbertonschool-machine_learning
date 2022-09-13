#!/usr/bin/env python3
"""
module which contains save_config and load_config function
"""
import tensorflow.keras as K


def save_config(network, filename):
    """
    saves a model’s configuration in JSON format:

    * network is the model whose weights should be saved
    * filename is the path of the file that the weights should be saved to
    * save_format is the format in which the weights should be saved
    Returns: None
    """

    filename += ".json" if ".json" not in filename else ''

    with open(filename, 'w') as f:
        f.write(network.to_json())


def load_config(filename):
    """
    loads a model with a specific configuration:

    * filename is the path of the file containing the model’s
      configuration in JSON format
    Returns: the loaded model
    """
    with open(filename, 'r') as f:
        loaded_model_json = f.read()
        return K.models.model_from_json(loaded_model_json)
