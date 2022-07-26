#!/usr/bin/env python3
"""
contains matrix_shape
"""


def matrix_shape(matrix):
    """
    calculates the shape of a matrix without using shape method from numpy
    """

    ls = []
    while (type(matrix) is list): ls.append(len(matrix)); matrix = matrix[0]
    return ls
