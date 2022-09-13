#!/usr/bin/env python3
"""
module which contains train_model function
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, verbose=True, shuffle=False):
    """
    Based on 4-train, updates the function to also analyze validaiton data:

    * validation_data is the data to validate the model with, if not None
    """
    return network.fit(x=data, y=labels, epochs=epochs,
                       batch_size=batch_size, verbose=verbose,
                       shuffle=shuffle, validation_data=validation_data)
