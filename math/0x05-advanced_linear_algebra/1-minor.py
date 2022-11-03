#!/usr/bin/env python3
"""
module which contains minor function
"""
determinant = __import__('0-determinant').determinant


def minor(matrix):
    """
    given a matrix, calculates its minor
    - matrix is a list of lists whose minor matrix should be calculated
    Returns: the minor matrix of matrix
    """

    if type(matrix) is not list:
        raise TypeError('matrix must be a list of lists')

    if not all([type(i) is list for i in matrix]):
        raise TypeError('matrix must be a list of lists')

    n = len(matrix)

    if n == 0:
        raise ValueError('matrix must be a non-empty square matrix')

    if matrix[0] and n not in [len(i) for i in matrix]:
        raise ValueError('matrix must be a non-empty square matrix')

    if matrix == [[]]:
        raise ValueError('matrix must be a non-empty square matrix')

    if n == 1:
        return [[1]]

    minors = []

    for x in range(n):
        min = []
        for y in range(n):
            det = []
            for i in range(n):
                nested = []
                if i == x:
                    continue
                for j in range(n):
                    if j == y:
                        continue
                    nested.append(matrix[i][j])
                det.append(nested)
            min.append(determinant(det))
        minors.append(min)
    return minors
