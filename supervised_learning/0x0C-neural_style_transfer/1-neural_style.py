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
        tf.enable_eager_execution()

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
        scaled_image = tf.image.resize_bicubic(image, new_shape)
        scaled_image = tf.clip_by_value(scaled_image / 255, 0, 1)

        return scaled_image

    def load_model(self):
        """
        * creates the model used to calculate cost
        * the model should use the VGG19 Keras model as a base
        * the model’s input should be the same as the VGG19 input
        * the model’s output should be a list containing the outputs of
          the VGG19 layers listed in style_layers followed by content _layer
        * saves the model in the instance attribute model
        """
        # for more information you can check in this blog (the link is sliced)
        # pt 1 https://medium.com/tensorflow/neural-style-transfer-creating-
        # pt 2 art-with-deep-learning-using-tf-keras-and-eager-
        # pt 3 execution-7d541ac31398
        vgg = tf.keras.applications.vgg19.VGG19(
            include_top=False, weights='imagenet')

        x = vgg.input

        style_outputs = []
        content_output = None

        for layer in vgg.layers[1:]:
            if isinstance(layer, tf.keras.layers.MaxPooling2D):
                layer = tf.keras.layers.AveragePooling2D(
                    pool_size=layer.pool_size, strides=layer.strides,
                    padding=layer.padding, name=layer.name)
                x = layer(x)
            else:
                x = layer(x)
                if layer.name in self.style_layers:
                    style_outputs.append(layer.output)

                if layer.name == self.content_layer:
                    content_output = x
                layer.trainable = False

        outputs = style_outputs + [content_output]

        return tf.keras.models.Model(x, outputs)
