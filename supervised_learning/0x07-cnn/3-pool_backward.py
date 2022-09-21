#!/usr/bin/env python3
"""
module which contains pool_backward function
"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    performs back propagation over a pooling layer of a neural network:

    * dA is a numpy.ndarray of shape (m, h_new, w_new, c_new)
      containing the partial derivatives with respect
      to the output of the pooling layer
        - m is the number of examples
        - h_new is the height of the output
        - w_new is the width of the output
        - c is the number of channels
    * A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c)
      containing the output of the previous layer
        - h_prev is the height of the previous layer
        - w_prev is the width of the previous layer
    * kernel_shape is a tuple of (kh, kw) containing the size
      of the kernel for the pooling
        - kh is the kernel height
        - kw is the kernel width
    * stride is a tuple of (sh, sw) containing the strides for the pooling
        - sh is the stride for the height
        - sw is the stride for the width
    * mode is a string containing either max or avg, indicating whether
      to perform maximum or average pooling, respectively
    Returns: the partial derivatives
             with respect to the previous layer (dA_prev)
    """
    m, h, w, c = A_prev.shape
    _, dh, dw, _ = dA.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros(A_prev.shape)

    for img in range(m):
        for i in range(dh):
            x = i * sh
            for j in range(dw):
                y = j * sw
                for k in range(c):
                    if mode == 'max':
                        A_prev_slice = A_prev[img, x: x + kh, y: y + kw, k]
                        mask = (A_prev_slice == np.max(A_prev_slice))
                        res = mask * dA[img, i, j, k]
                        dA_prev[img, x: x + kh, y: y + kw, k] += res
                    else:
                        average_dA = dA[img, i, j, k] / (kw * kh)
                        res = np.ones((kh, kw)) * average_dA
                        dA_prev[img, x: x + kh, y: y + kw, k] += res
    return dA_prev
