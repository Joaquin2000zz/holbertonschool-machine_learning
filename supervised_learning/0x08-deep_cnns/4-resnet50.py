#!/usr/bin/env python3
"""
module which containts resnet50 function
"""
import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    builds the ResNet-50 architecture as described in the paper
    "Deep Residual Learning for Image Recognition (2015)":

    * You can assume the input data will have shape (224, 224, 3)
    * All convolutions inside and outside the blocks should be followed
    * by batch normalization along the channels axis and a rectified linear
    * activation (ReLU), respectively.
    * All weights should use he normal initialization

    Returns: the keras model
    """
    het_et_al = K.initializers.HeNormal()

    X = K.Input(shape=(224, 224, 3))

    x = K.layers.Conv2D(filters=64, kernel_size=(7, 7),
                        strides=(2, 2), kernel_initializer=het_et_al,
                        padding='valid')(X)
    x = K.layers.BatchNormalization(axis=3)(x)

    x = projection_block(x, [64, 64, 256], s=1)
    x = identity_block(x, [64, 64, 256])
    x = identity_block(x, [64, 64, 256])

    x = projection_block(x, [128, 128, 512], s=1)
    x = identity_block(x, [128, 128, 512])
    x = identity_block(x, [128, 128, 512])
    x = identity_block(x, [128, 128, 512])

    x = projection_block(x, [256, 256, 1024], s=1)
    x = identity_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])

    x = projection_block(x, [512, 512, 2048], s=1)
    x = identity_block(x, [512, 512, 2048])
    x = identity_block(x, [512, 512, 2048])

    x = K.layers.AveragePooling2D(pool_size=(2, 2), strides=(1, 1))(x)
    x = K.layers.Dense(1000, kernel_initializer=het_et_al,
                       activation='softmax')(x)

    return K.Model(X, x)
