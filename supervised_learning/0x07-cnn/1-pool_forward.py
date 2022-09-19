#!/usr/bin/env python3
"""
module which contains pool_forward function
"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    performs forward propagation over a pooling layer of a neural network:

    * A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
      containing the output of the previous layer
        - m is the number of examples
        - h_prev is the height of the previous layer
        - w_prev is the width of the previous layer
        - c_prev is the number of channels in the previous layer
    * kernel_shape is a tuple of (kh, kw)
      containing the size of the kernel for the pooling
        - kh is the kernel height
        - kw is the kernel width
    * stride is a tuple of (sh, sw) containing the strides for the pooling
        - sh is the stride for the height
        - sw is the stride for the width
    * mode is a string containing either max or avg, indicating whether to
      perform maximum or average pooling, respectively
    Returns: the output of the pooling layer
    """
    m, h, w, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    oh = int((((h - kh) / sh) + 1))
    ow = int((((w - kw) / sw) + 1))

    ret_dim = (m, oh, ow, c)
    conv = np.zeros(shape=ret_dim)

    for i in range(ret_dim[1]):
        x = sh * i
        for j in range(ret_dim[2]):
            y = sw * j

            img_slice = A_prev[:, x: x + kh, y: y + kw, :]
            if mode == 'max':
                conv[:, i, j, :] = np.amax(img_slice, axis=(1, 2))
            else:
                conv[:, i, j, :] = np.mean(img_slice, axis=(1, 2))
    return conv
