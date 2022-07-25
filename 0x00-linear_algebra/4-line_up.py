#!/usr/bin/env python3
"""
contains add_arrays function
"""

import numpy as np


def add_arrays(arr1, arr2):
    """
    returns matrix addition from two arrays
    """
    arr1 = np.array(arr1)
    arr2 = np.array(arr2)
    return list(arr1 + arr2) if arr1.shape == arr2.shape else None
