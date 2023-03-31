#!/usr/bin/env python3
"""
module which contains change_brightness function
"""
import tensorflow as tf


def change_brightness(image, max_delta):
    """
    randomply changes the brightness in an image
    @image: 3D tf.Tensor of shape (w, h, 3) to transform
    @max_delta: maxing amount the image should be brightened (or darkened)
    Returns: the transformed image
    """
    if not isinstance(image, tf.Tensor):
        raise TypeError('image must be a tf.Tensor')
    n = len(image.shape)
    if n != 3:
        msg = 'random_brightness accepts a 3D [height, width, chanels] tensor'
        raise TypeError(msg)
    if not isinstance(max_delta, float):
        raise TypeError('max_delta must be a float')
    return tf.image.random_brightness(image, max_delta)
