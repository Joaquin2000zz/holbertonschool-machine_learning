#!/usr/bin/env python3
"""
module which contains one_hot_encode() function
"""

import numpy as np


def one_hot_encode(Y, classes):
    """
    converts a numeric label vector into a one-hot matrix
    """
    try:
        if classes < np.unique(Y).shape[0]:
            return None
        M = np.zeros(shape=(classes, Y.shape[0]))
        for i in range(classes):
            M[Y[i]][i] = 1
        return M
    except Exception as e:
        return None
