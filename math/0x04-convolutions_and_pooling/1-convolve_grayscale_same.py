#!/usr/bin/env python3
"""
module which contains convolve_grayscale_same function
"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    performs a same convolution on grayscale images:

    * images is a numpy.ndarray with shape (m, h, w)
      containing multiple grayscale images
        - m is the number of images
        - h is the height in pixels of the images
        - w is the width in pixels of the images
    * kernel is a numpy.ndarray with shape (kh, kw)
      containing the kernel for the convolution
        - kh is the height of the kernel
        - kw is the width of the kernel
    * if necessary, the image should be padded with 0’s
    * You are only allowed to use two for loops;
      any other loops of any kind are not allowed
    Returns: a numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    ph = int(np.ceil((kh - 1) / 2))
    pw = int(np.ceil((kw - 1) / 2))

    conv = np.zeros(shape=(m, h, w))

    padded_img = np.pad(images,
                        pad_width=((0, 0), (ph, ph), (pw, pw)),
                        mode="constant")

    for i in range(h):
        for j in range(w):
            img_slice = padded_img[: , i: i + kh, j: j + kw]
            conv[: , i, j] = np.tensordot(img_slice, kernel)
    return conv
