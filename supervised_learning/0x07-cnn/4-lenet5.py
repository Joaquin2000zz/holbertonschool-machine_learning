#!/usr/bin/env python3
"""
module which contains lenet5 function
"""
import tensorflow.compat.v1 as tf


def lenet5(x, y):
    """
    builds a modified version of the LeNet-5
    architecture using tensorflow:

    * x is a tf.placeholder of shape (m, 28, 28, 1)
      containing the input images for the network
        - m is the number of images
    * y is a tf.placeholder of shape (m, 10)
      containing the one-hot labels for the network
    * The model should consist of the following layers in order:
        - Convolutional layer with 6 kernels of shape 5x5 with same padding
        - Max pooling layer with kernels of shape 2x2 with 2x2 strides
        - Convolutional layer with 16 kernels of shape 5x5 with valid padding
        - Max pooling layer with kernels of shape 2x2 with 2x2 strides
        - Fully connected layer with 120 nodes
        - Fully connected layer with 84 nodes
        - Fully connected softmax output layer with 10 nodes
    * All layers requiring initialization should initialize their kernels
      with the he_normal initialization method:
        tf.keras.initializers.VarianceScaling(scale=2.0)
    * All hidden layers requiring activation should use
      the relu activation function
    Returns:
        - a tensor for the softmax activated output
        - a training operation that utilizes Adam optimization
          (with default hyperparameters)
        - a tensor for the loss of the netowrk
        - a tensor for the accuracy of the network
    """

    # he_normal initialization method
    # kernel_initializer=variance
    variance = tf.keras.initializers.VarianceScaling(scale=2.0)

    # Padding the input to make it 32x32. Specification of LeNET
    padded_input = tf.pad(x, [[0, 0], [2, 2], [2, 2], [0, 0]], "CONSTANT")

    # Convolutional Layer #1
    # Has a default stride of 1
    # Output: 28 * 28 * 6
    conv1 = tf.layers.conv2d(
        inputs=padded_input,
        filters=6,  # Number of filters.
        kernel_size=5,  # Size of each filter is 5x5.
        padding="same",  # Same padding applied to the input.
        activation=tf.nn.relu,
        kernel_initializer=variance)

    # Pooling Layer #1
    # Sampling half the output of previous layer
    # Output: 14 * 14 * 6
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2
    # Output: 10 * 10 * 16
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=16,  # Number of filters
        kernel_size=5,  # Size of each filter is 5x5
        padding="valid",  # No padding
        activation=tf.nn.relu,
        kernel_initializer=variance)

    # Pooling Layer #2
    # Output: 5 * 5 * 16
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2, pool_size=[2, 2], strides=2)

    # Reshaping output into a single dimention array
    # for input to fully connected layer
    pool2_flat = tf.layers.Flatten()(pool2)

    # Fully connected layer #1: Has 120 neurons
    dense1 = tf.layers.dense(
        inputs=pool2_flat, units=120, activation=tf.nn.relu,
        kernel_initializer=variance)

    # Fully connected layer #2: Has 84 neurons
    dense2 = tf.layers.dense(
        inputs=dense1, units=84, activation=tf.nn.relu,
        kernel_initializer=variance)

    # Output layer, 10 neurons for each digit
    logits = tf.layers.dense(inputs=dense2, units=10)

    y_pred = tf.nn.softmax(logits)

    # Compute the cross-entropy loss
    soft = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,
                                                      labels=y)
    loss = tf.reduce_mean(soft)

    # Use adam optimizer to reduce cost
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss)

    # For testing and prediction
    correct_pred = tf.equal(tf.cast(y_pred, "float"), tf.cast(y, "float"))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, "float"))

    return y_pred, train_op, loss, accuracy
