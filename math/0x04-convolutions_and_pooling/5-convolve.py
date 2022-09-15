#!/usr/bin/env python3
"""
module which contains convolve function
"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    performs a convolution on images using multiple kernels:

    * images is a numpy.ndarray with shape (m, h, w, c)
      containing multiple images
        - m is the number of images
        - h is the height in pixels of the images
        - w is the width in pixels of the images
        - c is the number of channels in the image
    * kernels is a numpy.ndarray with shape (kh, kw, c, nc)
      containing the kernels for the convolution
        - kh is the height of a kernel
        - kw is the width of a kernel
        - nc is the number of kernels
    * padding is either a tuple of (ph, pw), ‘same’, or ‘valid’
        - if ‘same’, performs a same convolution
        - if ‘valid’, performs a valid convolution
        - if a tuple:
            + ph is the padding for the height of the image
            + pw is the padding for the width of the image
        - the image should be padded with 0’s
    * stride is a tuple of (sh, sw)
        - sh is the stride for the height of the image
        - sw is the stride for the width of the image
    * You are only allowed to use three for loops;
      any other loops of any kind are not allowed
    Returns: a numpy.ndarray containing the convolved images
    """
    m, h, w, _ = images.shape
    kh, kw, _, nc = kernels.shape
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

    ret_dim = (m, oh, ow, nc)
    conv = np.zeros(shape=ret_dim)

    padded_img = np.pad(images,
                        pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
                        mode="constant")

    for i in range(ret_dim[1]):
        x = sh * i
        for j in range(ret_dim[2]):
            y = sw * j

            img_slice = padded_img[:, x: x + kh, y: y + kw, :]

            for k in range(nc):
                conv[:, i, j, k] = np.sum(img_slice * kernels[k], axis=(1, 2, 3))
    return conv
