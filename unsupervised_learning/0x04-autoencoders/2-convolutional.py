#!/usr/bin/env python3
"""
module which contains autoencoder function
"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    creates a convolutional autoencoder:

    - input_dims is a tuple of integers containing the dimensions
      of the model input
    - filters is a list containing the number of filters for each
      convolutional layer in the encoder, respectively
      * the filters should be reversed for the decoder
    - latent_dims is a tuple of integers containing the dimensions
      of the latent space representation
    - Each convolution in the encoder should use a kernel size of (3, 3)
      with same padding and relu activation, followed by
      max pooling of size (2, 2)
    - Each convolution in the decoder, except for the last two, should use
      a filter size of (3, 3) with same padding and relu activation,
      followed by upsampling of size (2, 2)
      * The second to last convolution should instead use valid padding
      * The last convolution should have the same number of filters as
        the number of channels in input_dims with sigmoid
        activation and no upsampling
    Returns: encoder, decoder, auto
      * encoder is the encoder model
      * decoder is the decoder model
      * auto is the full autoencoder model
    - The autoencoder model should be compiled using adam optimization
      and binary cross-entropy loss
    """
    X = keras.Input(shape=input_dims)
    Y = X
    for f in filters:
        Y = keras.layers.Conv2D(filters=f, activation='relu',
                                kernel_size=(3, 3), padding='same')(Y)
        Y = keras.layers.MaxPool2D(pool_size=(2, 2), padding='same')(Y)

    encoder = keras.Model(X, Y)  # f(x)

    xD = keras.Input(shape=latent_dims)

    Y = xD
    n = len(filters) - 1
    for i, f in enumerate(reversed(filters)):
        Y = keras.layers.Conv2D(filters=f, activation='relu',
                                kernel_size=(3, 3),
                                padding='same' if i != n else 'valid')(Y)
        Y = keras.layers.UpSampling2D(size=(2, 2))(Y)
    Y = keras.layers.Conv2D(filters=input_dims[-1], activation='sigmoid',
                            kernel_size=(3, 3), padding='same')(Y)

    decoder = keras.Model(xD, Y)  # g(h)

    # by definition: autoencoder -> x = g(f(x))
    auto = keras.Model(X, decoder(encoder(X)))

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
