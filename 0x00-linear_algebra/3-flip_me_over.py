#!/usr/bin/env python3
"""
contains matrix_transpose
"""

import numpy as np


def matrix_transpose(matrix):
    """
    returns the transpose of a 2D matrix using transpose numpy method
    """
    return [list(item) for item in np.array(matrix).transpose()]
