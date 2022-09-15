#!/usr/bin/env python3
"""
module which contains convolve_grayscale_valid function
"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    that performs a valid convolution on grayscale images:

    * images is a numpy.ndarray with shape (m, h, w)
      containing multiple grayscale images
        - m is the number of images
        - h is the height in pixels of the images
        - w is the width in pixels of the images
    * kernel is a numpy.ndarray with shape (kh, kw) containing
      the kernel for the convolution
        - kh is the height of the kernel
        - kw is the width of the kernel
    * You are only allowed to use two for loops; any other loops
      of any kind are not allowed
    Returns: a numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    ret_dim = (m, h - kh + 1, w - kw + 1)

    conv = np.zeros(shape=ret_dim)

    for i in range(ret_dim[1]):
        for j in range(ret_dim[2]):
            img_slice = images[:, i: i + kh, j: j + kw]
            conv[:, i, j] = np.tensordot(img_slice, kernel)
    return conv
