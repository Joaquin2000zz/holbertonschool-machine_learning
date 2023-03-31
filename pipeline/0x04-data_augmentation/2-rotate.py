#!/usr/bin/env python3
"""
module which contains rotate_image function
"""
import tensorflow as tf


def rotate_image(image, k: int=1):
    """
    rotates an image 90 degrees counter-clockwise
    @image: 3D tf.Tensor of shape (w, h, 3) to transform
    Returns: the transformed image
    """
    if not isinstance(image, tf.Tensor):
        raise TypeError('image must be a tf.Tensor')
    n = len(image.shape)
    if n != 3 and n != 4:
        msg = 'flip_left_right accepts a '
        msg += '3D [height, width, chanels] tensor'
        raise TypeError(msg)
    return tf.image.rot90(image, k=k)
