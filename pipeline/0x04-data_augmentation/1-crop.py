#!/usr/bin/env python3
"""
module which contains the crop_image function
"""
import tensorflow as tf


def crop_image(image, size):
    """
    makes a random crop in given an image and size
    @image: 3D tf.Tensor of shape (w, h, 3) containing the image to transform
    @size: tuple of shape (w, h)
    Returns: the transformed image
    """
    if not isinstance(image, tf.Tensor):
        raise TypeError('image must be a tf.Tensor')
    n = len(image.shape)
    if n != 3 and n != 4:
        msg = 'random_crop accepts a 3D [height, width, chanels]'
        raise TypeError(msg)
    n = len(size)
    if n != 3:
        raise TypeError('size must be a 1D tensor length 3')
    return tf.image.random_crop(image, size)
