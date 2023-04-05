#!/usr/bin/env python3
"""
module which contains the flip_image function
"""
import tensorflow as tf


def flip_image(image: tf.Tensor) -> tf.Tensor:
    """
    rotates an image horizontally
    @image: 3D tf.Tensor of shape (w, h, 3) containing the image to flip
    Returns: the transformed image
    """
    if not isinstance(image, tf.Tensor):
        raise TypeError('image must be a tf.Tensor')
    n = len(image.shape)
    if n != 3 and n != 4:
        msg = 'flip_left_right accepts whether:'
        msg += '\n3D [height, width, chanels] '
        msg += 'or 4D [batch, height, width, chanels] tensor'
        raise TypeError(msg)
    return tf.image.flip_left_right(image)
