#!/usr/bin/env python3
"""
module which contains inverse function
"""
compute_adjugate = __import__('3-adjugate').adjugate
compute_determinant = __import__('0-determinant').determinant


def inverse(matrix):
    """
    given a matrix computes its inverse
    - matrix is a list of lists whose inverse should be calculated
    Returns: the inverse of matrix, or None if matrix is singular
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

    determinant = compute_determinant(matrix)

    if determinant == 0:
        return None

    adjugate = compute_adjugate(matrix)

    for i in range(n):
        for j in range(n):
            adjugate[i][j] = 1 / determinant * adjugate[i][j]
    return adjugate
