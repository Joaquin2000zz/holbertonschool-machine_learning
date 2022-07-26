#!/usr/bin/env python3
"""
contains matrix_transpose
"""


from turtle import width


def matrix_transpose(matrix):
    """
    returns the transpose of a 2D matrix without using transpose numpy method
    """
    height = len(matrix)
    width = len(matrix[0])
    transposed = []

    for j in range(width):
        child = []
        for i in range(height):
            child.append(matrix[i][j])
        transposed.append(child)
    return transposed
