#!/usr/bin/env python3
"""
module which contains autoencoder function
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """
    creates a sparse autoencoder:

    - input_dims is an integer containing the dimensions of the model input
    - hidden_layers is a list containing the number of nodes for each hidden
      layer in the encoder, respectively
      * the hidden layers should be reversed for the decoder
    - latent_dims is an integer containing the dimensions of the
      latent space representation
    - lambtha is the regularization parameter used for L1 regularization
      on the encoded output
    Returns: encoder, decoder, auto
      * encoder is the encoder model
      * decoder is the decoder model
      * auto is the sparse autoencoder model
    - The sparse autoencoder model should be compiled using adam optimization
      and binary cross-entropy loss
    - All layers should use a relu activation except for the last layer
      in the decoder, which should use sigmoid
    """

    l1 = keras.regularizers.L1(lambtha)
    X = keras.Input(shape=(input_dims,))
    Y = X
    for l in hidden_layers:
        Y = keras.layers.Dense(units=l, activation='relu',
                               kernel_regularizer=l1)(Y)

    # bottle neck
    Y = keras.layers.Dense(units=latent_dims, activation='relu',
                           kernel_regularizer=l1)(Y)

    encoder = keras.Model(X, Y)  # f(x)

    xD = keras.Input(shape=(latent_dims,))

    Y = xD
    for l in reversed(hidden_layers):
        Y = keras.layers.Dense(units=l, activation='relu',
                               kernel_regularizer=l1)(Y)
    Y = keras.layers.Dense(units=input_dims, activation='sigmoid',
                           kernel_regularizer=l1)(Y)

    decoder = keras.Model(xD, Y)  # g(h)

    # by definition: autoencoder -> x = g(f(x))
    auto = keras.Model(X, decoder(encoder(X)))

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
