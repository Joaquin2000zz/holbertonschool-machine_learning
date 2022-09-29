#!/usr/bin/env python3
"""
module which containts dense_block function
"""
import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    builds the DenseNet-121 architecture as described in the paper
    "Densely Connected Convolutional Networks":

    * growth_rate is the growth rate
    * compression is the compression factor
    * You can assume the input data will have shape (224, 224, 3)
    * All convolutions should be preceded by Batch Normalization and
      a rectified linear activation (ReLU), respectively
    * All weights should use he normal initialization
    * Returns: the keras model
    """
    het_et_al = K.initializers.HeNormal()
    nb_filters = growth_rate * 2
    X = K.Input(shape=(224, 224, 3))

    x = K.layers.BatchNormalization()(X)
    x = K.layers.Activation('relu')(x)
    x = K.layers.Conv2D(nb_filters, kernel_size=(7, 7),
                        kernel_initializer=het_et_al,
                        strides=(2, 2), padding='same')(x)
    x = K.layers.MaxPool2D(pool_size=(3, 3), padding='same', strides=(2, 2))(x)

    for i, layers in enumerate([6, 12, 24, 16]):
        x, nb_filters = dense_block(x, nb_filters, growth_rate, layers)
        if i != 3:
            x, nb_filters = transition_layer(x, nb_filters, compression)

    x = K.layers.AveragePooling2D(pool_size=(7, 7), padding='same')(x)
    x = K.layers.Dense(1000, activation='softmax', kernel_initializer=het_et_al)(x)
    return K.Model(X, x)
