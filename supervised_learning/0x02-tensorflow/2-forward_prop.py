#!/usr/bin/env python3
"""
module which contains forward_prop
"""
import tensorflow.compat.v1 as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    creates the forward propagation graph for the neural network
    x is the placeholder for the input data
    layer_sizes is a list containing the number of nodes in each
    layer of the network
    activations is a list containing the activation functions
    for each layer of the network
    Returns: the prediction of the network in tensor form
    """

    for size, activation in zip(layer_sizes, activations):
        x = create_layer(x, size, activation)
    return x
