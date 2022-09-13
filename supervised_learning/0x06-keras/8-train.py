#!/usr/bin/env python3
"""
module which contains train_model
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                save_best=False, filepath=None,
                verbose=True, shuffle=False):
    """
    Based on 7-train, updates the function to also
    save the best iteration of the model:

    * save_best is a boolean indicating whether to save the model after
      each epoch if it is the best
        - a model is considered the best if its validation loss is
          the lowest that the model has obtained
    * filepath is the file path where the model should be saved
    """
    callbacks = []

    if early_stopping:
        early_stopping = K.callbacks.EarlyStopping(patience=patience,
                                                   monitor='val_loss')
        callbacks.append(early_stopping)

    if learning_rate_decay and validation_data:
        def s(global_step):
            """schedule performing inverse time decay"""
            return alpha / (1 + decay_rate * global_step)

        learning_rate_decay = K.callbacks.LearningRateScheduler(schedule=s,
                                                                verbose=1)
        callbacks.append(learning_rate_decay)

    if save_best:
        save_best = K.callbacks.ModelCheckpoint(filepath=filepath,
                                                save_best_only=True)
        callbacks.append(save_best)

    return network.fit(x=data, y=labels, epochs=epochs,
                       batch_size=batch_size, verbose=verbose,
                       shuffle=shuffle, validation_data=validation_data,
                       callbacks=[callbacks])
