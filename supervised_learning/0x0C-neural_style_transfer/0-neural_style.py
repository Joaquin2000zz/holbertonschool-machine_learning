#!/usr/bin/env python3
"""
module which contains the class NST
"""
import numpy as np
import tensorflow as tf


class NST:
    """
    class which performs the neural style transfer technique
    """
    style_layers = ['block1_conv1', 'block2_conv1',
                    'block3_conv1', 'block4_conv1', 'block5_conv1']

    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Class constructor:
        * style_image - the image used as a style reference,
          stored as a numpy.ndarray
        * content_image - the image used as a content reference,
          stored as a numpy.ndarray
        * alpha - the weight for content cost
        * beta - the weight for style cost
        * Sets Tensorflow to execute eagerly
        * Sets the instance attributes:
            - style_image: the preprocessed style image
            - content_image: the preprocessed content image
            - alpha: the weight for content cost
            - beta: the weight for style cost
        """
        error = 'must be a numpy.ndarray with shape (h, w, 3)'

        ndim = style_image.ndim
        shape = style_image.shape[2]
        if type(style_image) != np.ndarray or ndim != 3 or shape != 3:
            raise TypeError('style_image {}'.format(error))

        ndim = content_image.ndim
        shape = content_image.shape[2]
        if type(content_image) != np.ndarray or ndim != 3 or shape != 3:
            raise TypeError('content_image {}'.format(error))

        if (type(alpha) != int and type(alpha) != float) or alpha < 0:
            raise TypeError('alpha must be a non-negative number')

        if (type(beta) != int and type(beta) != float) or beta < 0:
            raise TypeError('beta must be a non-negative number')

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def scale_image(image):
        """
        rescales an image such that its pixels values are between
        0 and 1 and its largest side is 512 pixels
        * image - a numpy.ndarray of shape (h, w, 3) containing
          the image to be scaled
        * The scaled image should be a tf.tensor with the shape
          (1, h_new, w_new, 3) where max(h_new, w_new) == 512 and
          min(h_new, w_new) is scaled proportionately
        * The image should be resized using bicubic interpolation
        * After resizing, the image’s pixel values should be rescaled
          from the range [0, 255] to [0, 1].
        Returns: the scaled image
        """
        error = 'image must be a numpy.ndarray with shape (h, w, 3)'
        ndim = image.ndim
        shape = image.shape[2]
        if type(image) != np.ndarray or ndim != 3 or shape != 3:
            raise TypeError("{}".format(error))

        # calculating rescaling
        h, w, _ = image.shape
        max_dim = 512
        maximum = max(h, w)
        scale = max_dim / maximum
        new_shape = (int(h * scale), int(w * scale))
        image = np.expand_dims(image, axis=0)
        scaled_image = tf.image.resize(image, new_shape)
        scaled_image = tf.clip_by_value(scaled_image / 255, 0, 1)

        return scaled_image
