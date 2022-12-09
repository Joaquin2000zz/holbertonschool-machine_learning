#!/usr/bin/env python3
"""
module which contains autoencoder function
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    creates a variational autoencoder:

    - input_dims is an integer containing the dimensions of the model input
    - hidden_layers is a list containing the number of nodes for each
      hidden layer in the encoder, respectively
      * the hidden layers should be reversed for the decoder
    - latent_dims is an integer containing the dimensions of the latent
      space representation
    Returns: encoder, decoder, auto
      * encoder is the encoder model, which should output the latent
        representation, the mean, and the log variance, respectively
      * decoder is the decoder model
      * auto is the full autoencoder model
    - The autoencoder model should be compiled using adam optimization
      and binary cross-entropy loss
    - All layers should use a relu activation except for the mean and
      log variance layers in the encoder, which should use None, and the
      last layer in the decoder, which should use sigmoid
    """
    # building encoder f(x)
    X = keras.Input(shape=(input_dims,))
    Y = X
    for l in hidden_layers:
        Y = keras.layers.Dense(units=l, activation='relu')(Y)
    # now the μ and log(σ)
    mean = keras.layers.Dense(latent_dims)(Y)
    log_stddev = keras.layers.Dense(latent_dims)(Y)

    def sampler(args):
        """
        we sample from the standard normal a matrix
        of batch_size * latent_size (taking into account minibatches)
        """
        mean, log_stddev = args
        mean_shape = keras.backend.shape(mean)[0]
        dims = keras.backend.int_shape(mean)[1]
        std_norm = keras.backend.random_normal(shape=(mean_shape, dims),
                                               mean=0, stddev=1)
        # sampling from Z~N(μ, σ^2) is the same as
        # sampling from μ + σX, X~N(0,1)
        return mean + keras.backend.exp(log_stddev / 2) * std_norm
    # latent vector
    Z = keras.layers.Lambda(function=sampler,
                            output_shape=(latent_dims,))([mean, log_stddev])
    # encoder f(x)
    encoder = keras.Model(X, outputs=[Z, mean, log_stddev])

    # building decoder
    xD = keras.Input(shape=(latent_dims,))
    Y = xD
    for l in reversed(hidden_layers):
        Y = keras.layers.Dense(units=l, activation='relu')(Y)
    Y = keras.layers.Dense(units=input_dims, activation='sigmoid')(Y)
    decoder = keras.Model(xD, Y)  # decoder g(h)

    # creating autoencoder
    def vae_loss(x, y):
        """
        calculates variational autoencoder loss
        """
        bin_loss = keras.backend.binary_crossentropy(x, y)
        recon_loss = keras.backend.sum(bin_loss, axis=-1)
        # computes Kullback-Leibler divergece loss function
        exp_stddev = keras.backend.exp(log_stddev)
        sigma = keras.backend.mean(1 + log_stddev -
                                   keras.backend.square(mean) -
                                   keras.backend.square(exp_stddev),
                                   axis=-1)
        kl_loss = -0.5 * sigma
        total_loss = recon_loss + kl_loss
        return total_loss
    # by definition: decoder -> x = g(f(x))
    auto = keras.Model(X, decoder(encoder(X)[-1]))
    # the encoder(X)[-1] is because just takes log_stddev
    # to make the backpropagation

    auto.compile(optimizer='adam', loss=vae_loss)

    return encoder, decoder, auto
