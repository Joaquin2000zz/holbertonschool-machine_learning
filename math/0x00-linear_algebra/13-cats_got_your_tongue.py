#!/usr/bin/env python3
"""
contains np_cat
"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    concatenates two matrices along specific axis using concatenate function
    """
    return np.concatenate((mat1, mat2), int(axis))
