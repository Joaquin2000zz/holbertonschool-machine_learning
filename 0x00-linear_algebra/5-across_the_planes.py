#!/usr/bin/env python3
"""
contains add_matrices2D function
"""

import numpy as np


def add_matrices2D(mat1, mat2):
    """
    returns matrix addition from two matrices
    """
    mat1 = np.array(mat1)
    mat2 = np.array(mat2)
    return [list(item) for item in list(mat1 + mat2)] if mat1.shape == mat2.shape else None
