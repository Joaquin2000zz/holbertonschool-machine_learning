#!/usr/bin/env python3
"""
module which contains change_hue function
"""
import tensorflow as tf


def change_hue(image, delta):
    """
    changes the hue in an image in a factor of delta
    @image: 3D tf.Tensor of shape (w, h, 3) to transform
    @delta: amount the image should change
    Returns: the flipped image
    """
    if not isinstance(image, tf.Tensor):
        raise TypeError('image must be a tf.Tensor')
    n = len(image.shape)
    if n != 3:
        msg = 'change_hue accepts a 3D [height, width, chanels] tensor'
        raise TypeError(msg)
    if not isinstance(delta, float):
        raise TypeError('delta must be a float')
    return tf.image.adjust_hue(image, delta)
