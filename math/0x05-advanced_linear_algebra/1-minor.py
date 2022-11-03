#!/usr/bin/env python3
"""
module which contains minor function
"""


def minimize_matrix(matrix, ignorej, ignorez, n):
    """
    minimize matrix
    return a minimized matrixin given axis
    """
    new = []
    for j in range(n):
        if j == ignorej:
            continue
        nested = []
        for z in range(0, n):
            if z == ignorez:
                continue
            else:
                nested.append(matrix[j][z])
        new.append(nested)
    return new


def det_2(matrix):
    """
    finds the determinant of a 2x2 matrix
    """
    ad = matrix[0][0] * matrix[1][1]
    bc = matrix[0][1] * matrix[1][0]
    return ad - bc


def minor(matrix):
    """
    given a matrix, calculates its minor
    - matrix is a list of lists whose minor matrix should be calculated
    Returns: the minor matrix of matrix
    """
    if not matrix or not isinstance(matrix, list):
        raise TypeError('matrix must be a list of lists')
    if not all([isinstance(i, list) for i in matrix]):
        raise TypeError('matrix must be a list of lists')

    n = len(matrix)

    if n == 1:
        if not matrix[0]:
            raise ValueError('matrix must be a non-empty square matrix')
        else:
            if len(matrix[0]) != 1:
                raise ValueError('matrix must be a non-empty square matrix')
            return [[1]]
    m = [len(i) for i in matrix]
    if n not in m:
        raise ValueError('matrix must be a non-empty square matrix')

    minors = []

    if n == 2:
        for i in range(n):
            j = 0
            new = []
            new.append(minimize_matrix(matrix, i, j, n)[0][0])
            for j in range(1, n):
                new.append(minimize_matrix(matrix, i, j, n)[0][0])
            minors.append(new)

        return minors

    for i in range(n):
        new = []
        for j in range(n):
            new.append(det_2(minimize_matrix(matrix, ignorej=i,
                             ignorez=j, n=n)))
        minors.append(new)

    return minors
