#!/usr/bin/env python3
"""
module which contains determinant function
"""


def determinant(matrix):
    """
    given a matrix, calculates its determinant
    - matrix is a list of lists whose determinant should be calculated
    - The list [[]] represents a 0x0 matrix
    Returns: its matrix's determinant
    """
    if not matrix or not isinstance(matrix, list):
        raise TypeError('matrix must be a list of lists')
    if not isinstance(matrix[:], list):
        raise TypeError('matrix must be a list of lists')

    n = len(matrix)

    if n == 1:
        if not matrix[0]:
            return 1
        else:
            if len(matrix[0]) != 1:
                raise ValueError('message matrix must be a square matrix')
            return matrix[0][0]

    m = [len(i) for i in matrix]
    if n not in m:
        raise ValueError('message matrix must be a square matrix')

    if n == 2:
        ad = matrix[0][0] * matrix[1][1]
        bc = matrix[0][1] * matrix[1][0]
        return ad - bc

    a, b, c = matrix[0][0], matrix[0][1], matrix[0][2]

    ei, fh = matrix[1][1] * matrix[2][2], matrix[1][2] * matrix[2][1]
    di, fg = matrix[1][0] * matrix[2][2], matrix[1][2] * matrix[2][0]
    dh, eg = matrix[1][0] * matrix[2][1], matrix[1][1] * matrix[2][0]

    return a * (ei - fh) - b * (di - fg) + c * (dh - eg)
