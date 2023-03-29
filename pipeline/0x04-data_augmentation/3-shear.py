#!/usr/bin/env python3
"""
module which contains shear_image function
"""
import tensorflow as tf


def shear_image(image, intensity):
    """
    randomply shears an image
    @image: 3D tf.Tensor of shape (w, h, 3) to transform
    @intensity: transformation intensity in degrees
    Returns: the transformed image
    """
    if not isinstance(image, tf.Tensor):
        raise TypeError('image must be a tf.Tensor')
    n = len(image.shape)
    if n != 3:
        msg = 'random_shear accepts 3D [height, width, chanels] '
        raise TypeError(msg)
    if not isinstance(intensity, int):
        raise TypeError('intensity must be an integer')
    return tf.keras.preprocessing.image.random_shear(image, intensity)
