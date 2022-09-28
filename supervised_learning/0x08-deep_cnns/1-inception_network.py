#!/usr/bin/env python3
"""
module which contains inception_network function
"""
import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    builds the inception network as described in the paper
    "Going Deeper with Convolutions (2014)":

    * You can assume the input data will have shape (224, 224, 3)
    * All convolutions inside and outside the inception block
      should use a rectified linear activation (ReLU)
    Returns: the keras model
    """
    het_et_al = K.initializers.HeNormal()

    X = K.Input(shape=(224, 224, 3))

    x = K.layers.Conv2D(64, kernel_size=(7, 7), padding='same',
                        strides=(2, 2), kernel_initializer=het_et_al, activation='relu')(X)
    x = K.layers.MaxPool2D((3, 3), padding='same', strides=(2, 2))(x)
    x = K.layers.Conv2D(64, kernel_size=(1, 1), padding='same',
                        strides=(1, 1), kernel_initializer=het_et_al, activation='relu')(x)
    x = K.layers.Conv2D(192, kernel_size=(3, 3), padding='same',
                        strides=(1, 1), kernel_initializer=het_et_al, activation='relu')(x)
    x = K.layers.MaxPool2D((3, 3), padding='same', strides=(2, 2))(x)

    x = inception_block(x, [64, 96, 128, 16, 32, 32])
    x = inception_block(x, [128, 128, 192, 32, 96, 64])

    x = K.layers.MaxPool2D((3, 3), padding='same', strides=(2, 2))(x)

    x = inception_block(x, [192, 96, 208, 16, 48, 64])
    x = inception_block(x, [160, 112, 224, 24, 64, 64])
    x = inception_block(x, [128, 128, 256, 24, 64, 64])
    x = inception_block(x, [112, 144, 288, 32, 64, 64])
    x = inception_block(x, [256, 160, 320, 32, 128, 128])

    x = K.layers.MaxPool2D((3, 3), padding='same', strides=(2, 2))(x)

    x = inception_block(x, [256, 160, 320, 32, 128, 128])
    x = inception_block(x, [384, 192, 384, 48, 128, 128])

    x = K.layers.AveragePooling2D((7, 7), (1, 1))(x)
    x = K.layers.Dropout(rate=0.4)(x)
    x = K.layers.Dense(1000, kernel_initializer=het_et_al, activation='softmax')(x)
    return K.Model(X, x)
