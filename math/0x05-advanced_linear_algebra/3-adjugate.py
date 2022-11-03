#!/usr/bin/env python3
"""
module which contains adjugate function
"""
compute_cofactor = __import__('2-cofactor').cofactor


def adjugate(matrix):
    """
    given a matrix, computes its adjugate
    - matrix is a list of lists whose adjugate matrix should be calculated
    Returns: the adjugate matrix of matrix
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

    cofactor = compute_cofactor(matrix)

    for i in range(n):
        for j in range(n):
            if j <= i:
                continue
            cofactor[i][j], cofactor[j][i] = cofactor[j][i], cofactor[i][j]
    return cofactor
