#!/usr/bin/env python3
"""
module which contains cofactor fuction
"""
compute_minor = __import__('1-minor').minor

def cofactor(matrix):
    """
    given a matrix, computes its minor
    Returns: the cofactor matrix of matrix
    """

    if type(matrix) is not list:
        raise TypeError('matrix must be a list of lists')

    if all([type(i) is list for i in matrix]) is False:
        raise TypeError('matrix must be a list of lists')

    n = len(matrix)

    if n == 0:
        raise TypeError('matrix must be a list of lists')

    if (matrix[0] and n != len(matrix[0])) or matrix == [[]]:
        raise ValueError('matrix must be a non-empty square matrix')

    if all(n == len(row) for row in matrix) is False:
        raise ValueError('matrix must be a non-empty square matrix')

    if n == len(matrix[0]) == 1:
        return [[1]]
    
    minor = compute_minor(matrix)

    sign = 1

    for i in range(n):
        for j in range(n):
            minor[i][j] = minor[i][j] * sign
            sign *= -1
    return minor
