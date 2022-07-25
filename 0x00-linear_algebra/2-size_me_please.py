#!/usr/bin/env python3
"""
contains matrix_shape
"""

import numpy as np


def matrix_shape(matrix):
    """
    calculates the shape of a matrix using shape method from numpy
    """
    return list(np.array(matrix).shape)
