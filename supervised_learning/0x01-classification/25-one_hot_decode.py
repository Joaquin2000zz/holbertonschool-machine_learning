#!/usr/bin/env python3
"""
module which contains one_hot_decode() function
"""

import numpy as np


def one_hot_decode(one_hot):
    """
    converts a one-hot matrix into a numeric label vector
    """
    try:
        return np.argmax(one_hot, axis=0)
    except Exception as e:
        return None
