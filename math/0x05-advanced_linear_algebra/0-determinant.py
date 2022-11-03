#!/usr/bin/env python3
"""
module which contains determinant function
"""


def minimize_matrix(matrix, ignore, n):
    """
    minimize matrix
    return a minimized matrixin given axis
    """
    new = []
    for j in range(1, n):
        nested = []
        for z in range(0, n):
            if z == ignore:
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


def determinant(matrix):
    """
    given a matrix, calculates its determinant
    - matrix is a list of lists whose determinant should be calculated
    - The list [[]] represents a 0x0 matrix
    Returns: its matrix's determinant
    """
    if type(matrix) is not list:
        raise TypeError('matrix must be a list of lists')

    if not all([type(i) is list for i in matrix]):
        raise TypeError('matrix must be a list of lists')

    n = len(matrix)

    if n == 0:
        raise TypeError('matrix must be a list of lists')

    if matrix[0] and n not in [len(i) for i in matrix]:
        raise ValueError('matrix must be a square matrix')

    if matrix == [[]]:
        return 1

    if len(matrix) == 1:
        return matrix[0][0]

    if all(len(matrix) == len(col) for col in matrix) is False:
        raise ValueError('matrix must be a square matrix')

    if n == 2:
        return det_2(matrix)

    det = 0
    sign = 1

    for m, k, i in zip(matrix, matrix[0], range(n)):
        det += sign * k * determinant(minimize_matrix(matrix, ignore=i, n=n))
        sign *= -1
    return det
