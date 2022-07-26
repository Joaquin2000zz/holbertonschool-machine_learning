#!/usr/bin/env python3
"""
contains add_arrays function
"""


def add_arrays(arr1, arr2):
    """
    returns matrix addition from two arrays without numpy
    """
    length = len(arr1)
    if length != len(arr2):
        return None
    add = []
    i = 0
    while (i < length): add.append(arr1[i] + arr2[i]); i += 1
    return add
