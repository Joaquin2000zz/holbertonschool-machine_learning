#!/usr/bin/env python3
"""
module which contains one_hot function
"""
import numpy as np
import tensorflow.keras as K

def one_hot(labels, classes=None):
    """
    converts a label vector into a one-hot matrix:

    * The last dimension of the one-hot matrix must be
      the number of classes
    Returns: the one-hot matrix
    """
    m = labels.shape[0]
    if not classes:
        classes = np.amax(labels) + 1
    one_hot = np.zeros((m , classes))
    one_hot[np.arange(m), labels] = 1
    return one_hot
