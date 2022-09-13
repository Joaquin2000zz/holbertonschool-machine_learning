#!/usr/bin/env python3
"""
module which contains train_model
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                verbose=True, shuffle=False):
    """
    Based on 5-train, updates the function to also train
    the model using early stopping:

    * early_stopping is a boolean that indicates whether early
      stopping should be used
        - early stopping should only be performed if validation_data exists
        - early stopping should be based on validation loss
    * patience is the patience used for early stopping
    """
    if early_stopping:
        early_stopping = K.callbacks.EarlyStopping(patience=patience)
    return network.fit(x=data, y=labels, epochs=epochs,
                       batch_size=batch_size, verbose=verbose,
                       shuffle=shuffle, validation_data=validation_data,
                       callbacks=[early_stopping])
