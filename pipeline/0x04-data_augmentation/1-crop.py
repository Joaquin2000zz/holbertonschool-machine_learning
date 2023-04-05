#!/usr/bin/env python3
"""
module which contains the crop_image function
"""
import tensorflow as tf


def crop_image(image, size: tuple=(200, 200, 3)) -> tf.Tensor:
    """
    makes a random crop in given an image and size
    @image: tf.Tensor of shape (w, h, 3) containing the image
    @size: tuple of shape (w, h)
    Returns: the cropped image
    """
    if not isinstance(image, tf.Tensor):
        raise TypeError('image must be a tf.Tensor')
    n = len(image.shape)
    if n != 3 and n != 4:
        msg = 'flip_left_right accepts whether:'
        msg += '\n3D [height, width, chanels] '
        msg += 'or 4D [batch, height, width, chanels] tensor'
        raise TypeError(msg)
    n = len(size)
    if n != 3:
        raise TypeError('size must be a 1D tensor length 3')
    try:
      return tf.image.random_crop(image, size)
    except:
      return False