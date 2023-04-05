#!/usr/bin/env python3
"""
module which contains change_brightness function
"""
import numpy as np
import tensorflow as tf


def change_brightness(image: tf.Tensor,
                      max_delta: float=np.random.uniform(0,
                                                         .3)) -> tf.Tensor:
    """
    randomply changes the brghtness in an image
    @image: 3D tf.Tensor of shape (w, h, 3) to transform
    @max_delta: maxing amount the image should be brightened (or darkened)
    Returns: the flipped image
    """
    if not isinstance(image, tf.Tensor):
        raise TypeError('image must be a tf.Tensor')
    n = len(image.shape)
    if n != 3:
        msg = 'random_brightness accepts 3D [height, width, chanels] '
        raise TypeError(msg)
    if not isinstance(max_delta, float):
        raise TypeError('max_delta must be a float')
    return tf.image.random_brightness(image, max_delta)
