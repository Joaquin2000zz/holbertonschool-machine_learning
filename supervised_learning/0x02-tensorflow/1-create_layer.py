"""
module which contains create_layer
"""
import tensorflow.compat.v1 as tf


def create_layer(prev, n, activation):
    """
    prev is the tensor output of the previous layer
    n is the number of nodes in the layer to create
    activation is the activation function that the layer should use
    use tf.keras.initializers.VarianceScaling(mode='fan_avg')
    to implement He-et.-al initialization for the layer weights
    each layer should be given the name layer
    Returns: the tensor output of the layer
    """

    #  implement He-et-al initialization for the layer weights
    het_et_al = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    # kernel_initializer=het_et_al
    linear_model = tf.layers.Dense(name="layer", units=n,
                                   activation=activation,
                                   kernel_initializer=het_et_al)
    layer = linear_model(prev)
    return layer