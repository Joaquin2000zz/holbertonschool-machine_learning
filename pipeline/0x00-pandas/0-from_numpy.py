#!/usr/bin/env python3
"""
module which contains from_numpy function
"""
import numpy as np
import pandas as pd


def from_numpy(array):
    """
    creates a pd.DataFrame from a np.ndarray:

    - array: is the np.ndarray from which you should create the pd.DataFrame
    The columns of the pd.DataFrame should be labeled in alphabetical order
    and capitalized. There will not be more than 26 columns.
    Returns: the newly created pd.DataFrame
    """
    if not isinstance(array, np.ndarray):
        raise TypeError('array must be an np.ndarray')
    n = array.shape[1]
    df = pd.DataFrame(array, columns=[chr(65 + i) for i in range(n)])
    return df
