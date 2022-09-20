#!/usr/bin/env python3
"""
module which contains conv_backward function
"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    performs back propagation over a convolutional layer of a neural network:

    * dZ is a numpy.ndarray of shape (m, h_new, w_new, c_new)
      containing the partial derivatives with respect to the
      unactivated output of the convolutional layer
        - m is the number of examples
        - h_new is the height of the output
        - w_new is the width of the output
        - c_new is the number of channels in the output
    * A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
      containing the output of the previous layer
        - h_prev is the height of the previous layer
        - w_prev is the width of the previous layer
        - c_prev is the number of channels in the previous layer
    * W is a numpy.ndarray of shape (kh, kw, c_prev, c_new)
      containing the kernels for the convolution
        - kh is the filter height
        - kw is the filter width
    * b is a numpy.ndarray of shape (1, 1, 1, c_new)
      containing the biases applied to the convolution
    * padding is a string that is either same or valid,
      indicating the type of padding used
    * stride is a tuple of (sh, sw) containing the strides for the convolution
        - sh is the stride for the height
        - sw is the stride for the width
    Returns: the partial derivatives with respect to the
    previous layer (dA_prev), the kernels (dW),
    and the biases (db), respectively
    """
    m, h, w, _ = A_prev.shape
    _, zh, zw, zc = dZ.shape
    kh, kw, _, _ = W.shape
    sh, sw = stride

    if padding == "same":
        ph = int(np.ceil(((h - 1) * sh + kh - h) / 2))
        pw = int(np.ceil(((w - 1) * sw + kw - w) / 2))
    else:
        ph = pw = 0

    padded_img = np.pad(A_prev,
                        pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
                        mode="constant")

    dA_prev = np.zeros(padded_img.shape)
    dW = np.zeros(W.shape)

    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    for img in range(m):
        for i in range(zh):
            x = i * sh
            for j in range(zw):
                y = j * sw
                for k in range(zc):

                    dz = dZ[img, i, j, k]
                    slice_img = padded_img[img, x: x + kh, y: y + kw, :]

                    dW[:, :, :, k] += dz * slice_img

                    dA_prev[img, x: x + kh, y: y + kw, :] += dz * W[:, :, :, k]

    dA_prev = dA_prev[: ,ph:-ph, pw:-pw, :] if padding == "same" else dA_prev

    return dA_prev, dW, db
