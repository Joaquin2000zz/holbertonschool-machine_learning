#!/usr/bin/env python3
"""
module which contains convolve function
"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    performs pooling on images:
    * images is a numpy.ndarray with shape (m, h, w, c)
      containing multiple images
        - m is the number of images
        - h is the height in pixels of the images
        - w is the width in pixels of the images
        - c is the number of channels in the image
    * kernel_shape is a tuple of (kh, kw) containing the kernel
      shape for the pooling
        - kh is the height of the kernel
        - kw is the width of the kernel
    * stride is a tuple of (sh, sw)
        - sh is the stride for the height of the image
        - sw is the stride for the width of the image
    * mode indicates the type of pooling
        - max indicates max pooling
        - avg indicates average pooling
    * You are only allowed to use two for loops; any other
      loops of any kind are not allowed
    Returns: a numpy.ndarray containing the pooled images
    """

    m, h, w , c= images.shape
    kh, kw = kernel_shape
    sh, sw = stride


    oh = int((((h  - kh) / sh) + 1))
    ow = int((((w - kw) / sw) + 1))

    ret_dim = (m, oh, ow, c)
    conv = np.zeros(shape=ret_dim)

    for i in range(ret_dim[1]):
        x = sh * i
        for j in range(ret_dim[2]):
            y = sw * j

            img_slice = images[:, x: x + kh, y: y + kw, : ]
            if mode == 'max':
                conv[:, i, j, : ] = np.amax(img_slice, axis=0)
            else:
                conv[:, i, j] = np.mean(img_slice, axis=0)
    return conv
