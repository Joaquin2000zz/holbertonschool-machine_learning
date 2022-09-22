#!/usr/bin/env python3
"""
module which contains lenet5 function
"""
import tensorflow.keras as K


def lenet5(X):
    """
    builds a modified version of the LeNet-5 architecture using keras:

    * X is a K.Input of shape (m, 28, 28, 1)
      containing the input images for the network
        - m is the number of images
    * The model should consist of the following layers in order:
    * Convolutional layer with 6 kernels of shape 5x5 with same padding
    * Max pooling layer with kernels of shape 2x2 with 2x2 strides
    * Convolutional layer with 16 kernels of shape 5x5 with valid padding
    * Max pooling layer with kernels of shape 2x2 with 2x2 strides
    * Fully connected layer with 120 nodes
    * Fully connected layer with 84 nodes
    * Fully connected softmax output layer with 10 nodes
    * All layers requiring initialization should initialize their kernels
      with the he_normal initialization method
    * All hidden layers requiring activation should use
      the relu activation function
    Returns: a K.Model compiled to use Adam optimization
             (with default hyperparameters) and accuracy metrics
    """
    # he_normal initialization method
    # kernel_initializer=variance
    variance = K.initializers.VarianceScaling(scale=2.0)

    # Convolutional Layer #1
    # Has a default stride of 1
    # Output: 28 * 28 * 6
    conv1 = K.layers.Conv2D(
        filters=6,  # Number of filters.
        kernel_size=5,  # Size of each filter is 5x5.
        padding="same",  # Same padding applied to the input.
        activation=K.activations.relu,
        kernel_initializer=variance)(X)

    # Pooling Layer #1
    # Sampling half the output of previous layer
    # Output: 14 * 14 * 6
    pool1 = K.layers.MaxPool2D(pool_size=[2, 2], strides=2)(conv1)

    # Convolutional Layer #2
    # Output: 10 * 10 * 16
    conv2 = K.layers.Conv2D(
        filters=16,  # Number of filters
        kernel_size=5,  # Size of each filter is 5x5
        padding="valid",  # No padding
        activation=K.activations.relu,
        kernel_initializer=variance)(pool1)

    # Pooling Layer #2
    # Output: 5 * 5 * 16
    pool2 = K.layers.MaxPool2D(pool_size=[2, 2], strides=2)(conv2)

    # Reshaping output into a single dimention array
    # for input to fully connected layer
    pool2_flat = K.layers.Flatten()(pool2)

    # Fully connected layer #1: Has 120 neurons
    dense1 = K.layers.Dense(
        units=120, activation=K.activations.relu,
        kernel_initializer=variance)(pool2_flat)

    # Fully connected layer #2: Has 84 neurons
    dense2 = K.layers.Dense(
        units=84, activation=K.activations.relu,
        kernel_initializer=variance)(dense1)

    # Output layer, 10 neurons for each digit
    logits = K.layers.Dense(units=10,
                            kernel_initializer=variance)(dense2)

    y_pred = K.activations.softmax(logits)

    network = K.Model(X, y_pred)
    # Use adam optimizer to reduce cost
    optimizer = K.optimizers.Adam()
    network.compile(optimizer=optimizer,
                    loss='categorical_crossentropy')

    return network
