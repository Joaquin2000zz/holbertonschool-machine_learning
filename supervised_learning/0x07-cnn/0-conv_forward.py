#!/usr/bin/env python3
"""
module which contains conv_forward function
"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    performs forward propagation over a convolutional
    layer of a neural network:

    * A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
      containing the output of the previous layer
        - m is the number of examples
        - h_prev is the height of the previous layer
        - w_prev is the width of the previous layer
        - c_prev is the number of channels in the previous layer
    * W is a numpy.ndarray of shape (kh, kw, c_prev, c_new)
      containing the W for the convolution
        - kh is the filter height
        - kw is the filter width
        - c_prev is the number of channels in the previous layer
        - c_new is the number of channels in the output
    * b is a numpy.ndarray of shape (1, 1, 1, c_new)
      containing the biases applied to the convolution
    * activation is an activation function applied to the convolution
    * padding is a string that is either same or valid,
      indicating the type of padding used
    * stride is a tuple of (sh, sw) containing the strides for the convolution
        - sh is the stride for the height
        - sw is the stride for the width
    Returns: the output of the convolutional layer
    """
    m, h, w, _ = A_prev.shape
    kh, kw, _, nc = W.shape
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

    padded_img = np.pad(A_prev,
                        pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
                        mode="constant")

    for i in range(ret_dim[1]):
        x = sh * i
        for j in range(ret_dim[2]):
            y = sw * j

            img_slice = padded_img[:, x: x + kh, y: y + kw, :]

            for k in range(nc):
                conv[:, i, j, k] = np.sum(
                    img_slice * W[:, :, :, k], axis=(1, 2, 3))
    return activation(conv + b)
