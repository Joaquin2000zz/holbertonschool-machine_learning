#!/usr/bin/env python3
"""
module which contains convolve_grayscale function
"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
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
    * padding is either a tuple of (ph, pw), ‘same’, or ‘valid’
        - if ‘same’, performs a same convolution
        - if ‘valid’, performs a valid convolution
        - if is a tuple of (ph, pw)
            + ph is the padding for the height of the image
            + pw is the padding for the width of the image
            + the image should be padded with 0’s
    * You are only allowed to use two for loops;
      any other loops of any kind are not allowed
    * stride is a tuple of (sh, sw)
        - sh is the stride for the height of the image
        - sw is the stride for the width of the image
    Returns: a numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    if padding == "same":
        ph = int(np.ceil(((h - 1) * sh + kh - h) / 2))
        pw = int(np.ceil(((w - 1) * sw + kw - w) / 2))
    elif padding == "valid":
        ph = pw = 0
    else:
        ph, pw = padding

    oh = int((((h + 2 * ph - kh) / sh) + 1))
    ow = int((((w + 2 * pw - kw) / sw) + 1))

    ret_dim = (m, oh, ow)
    conv = np.zeros(shape=ret_dim)

    padded_img = np.pad(images,
                        pad_width=((0, 0), (ph, ph), (pw, pw)),
                        mode="constant")

    for i in range(ret_dim[1]):
        x = sh * i
        for j in range(ret_dim[2]):
            y = sw * j

            img_slice = padded_img[:, x: x + kh, y: y + kw]
            conv[:, i, j] = np.tensordot(img_slice, kernel)
    return conv
