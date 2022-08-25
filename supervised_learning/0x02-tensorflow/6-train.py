#!/usr/bin/env python3
"""
module which contains calculate_loss
"""
import tensorflow.compat.v1 as tf
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes,
          activations, alpha, iterations, save_path="/tmp/model.ckpt"):
    """
    that builds, trains, and saves a neural network classifier:

    X_train is a numpy.ndarray containing the training input data
    Y_train is a numpy.ndarray containing the training labels
    X_valid is a numpy.ndarray containing the validation input data
    Y_valid is a numpy.ndarray containing the validation labels
    layer_sizes is a list containing the number of nodes in each layer of the network
    activations is a list containing the activation functions for each layer of the network
    alpha is the learning rate
    iterations is the number of iterations to train over
    save_path designates where to save the model
    Add the following to the graph’s collection
    placeholders x and y
    tensors y_pred, loss, and accuracy
    operation train_op
    """
    x, y = create_placeholders(X_train.shape[1],
                               Y_train.shape[1])

    y_pred = forward_prop(x, layer_sizes,
                              activations)

    accuracy = calculate_accuracy(Y_train, y)
    
    loss = calculate_loss(y, activation)
    
    train = create_train_op(loss, alpha)
    
    init = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(init)



        for _ in range(iterations):
            _, loss_value = session.run((train, loss))
            print(loss_value)
