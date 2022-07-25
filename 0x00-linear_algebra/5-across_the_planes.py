#!/usr/bin/env python3
"""
contains add_matrices2D function
"""

import numpy as np


def add_matrices2D(mat1, mat2):
    """
    returns matrix addition from two matrices
    """
    m1 = np.array(mat1)
    m2 = np.array(mat2)
    return [list(
            item) for item in list(
                              m1 + m2)] if m1.shape == m2.shape else None
