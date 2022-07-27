#!/usr/bin/env python3
"""
contains add_matrices 2D function
"""

def add_matrices2D(mat1, mat2):
    """
    returns matrix addition from two 2D matrices without numpy
    """
    height = len(mat1)
    add = []
    for i in range(height):
        width = len(mat1[i])
        if width != len(mat2[i]):
            return None
        child = []
        j = 0
        while j < width:
            child.append(mat1[i][j] + mat2[i][j])
            j += 1
        add.append(child)
    return add
