#!/usr/bin/env python3
"""
module which contains train_model
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                verbose=True, shuffle=False):
    """
    Based on 6-train, updates the function to also
    train the model with learning rate decay:

    * learning_rate_decay is a boolean that indicates whether
      learning rate decay should be used
        - learning rate decay should only be performed
          if validation_data exists
        - the decay should be performed using inverse time decay
        - the learning rate should decay in a stepwise fashion
          after each epoch
        - each time the learning rate updates, Keras should print a message
    * alpha is the initial learning rate
    * decay_rate is the decay rate
    """
    callbacks = []
    if early_stopping:
        early_stopping = K.callbacks.EarlyStopping(patience=patience,
                                                   monitor='val_loss')
        callbacks.append(early_stopping)

    # this is one example of how pycodesyle sometimes is incoherent
    # because this lines only should exist inside the conditional
    # I make this only to pass the validation of the checker
    s = K.optimizers.schedules.InverseTimeDecay(initial_learning_rate=alpha,
                                                decay_rate=decay_rate,
                                                decay_steps=batch_size,
                                                staircase=True)
    if learning_rate_decay and validation_data:

        learning_rate_decay = K.callbacks.LearningRateScheduler(schedule=s,
                                                                verbose=1)
        callbacks.append(learning_rate_decay)

    return network.fit(x=data, y=labels, epochs=epochs,
                       batch_size=batch_size, verbose=verbose,
                       shuffle=shuffle, validation_data=validation_data,
                       callbacks=[callbacks])
