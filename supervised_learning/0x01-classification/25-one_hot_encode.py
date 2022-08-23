#!/usr/bin/env python3
"""
module which contains one_hot_encode() function
"""

import numpy as np


def one_hot_encode(Y, classes):
    """
    converts a numeric label vector into a one-hot matrix
    """
    M = np.zeros(shape=(Y.shape[0], classes))
    for i in range(classes):
        M[Y[i]][i] = 1
    return M
