#!/usr/bin/env python3
"""
module which contains cat_matrices2D
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    cocatenates two 2D matrix
    """

    mat1copy = list(map(list, mat1))
    if not axis:
        for item in mat2:
            mat1copy.append(item)
        return mat1copy
    if axis == 1:
        width = len(mat1copy[0]) if type(mat1copy) is list else None
        n = len(mat2)
        if width == n:
            i = 0
            for item in mat2:
                
                mat1copy[i] = mat1copy[i]
                mat1copy[i] = mat1copy[i] + item
                i += 1
            return mat1copy

        return None
    else:
        return None
