#!/usr/bin/env python3
"""
module which contains determinant function
"""


def minimize_matrix(matrix, n):
    """
    minimize matrix
    return a list of minimized matrices and other with k values
    """
    det = 0
    minimized = []
    for i in range(0, n):
        new = []
        for j in range(1, n):
            nested = []
            for z in range(0, n):
                if z == i:
                    continue
                else:
                    nested.append(matrix[j][z])
            new.append(nested)
        minimized.append(new)
    return minimized, matrix[0]


def det_2(matrix):
    """
    finds the determinant of a 2x2 matrix
    """
    ad = matrix[0][0] * matrix[1][1]
    bc = matrix[0][1] * matrix[1][0]
    return ad - bc


def det_3(minimized, kvector):
    """
    finds the determinant of a minimized 3x3 matrix
    """
    sign = 1
    det = 0

    for min, k in zip(minimized, kvector):
        det += k * det_2(min) * sign
        sign *= -1
    return det


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
        return det_2(matrix)

    minimized, kvector = minimize_matrix(matrix, n)
    if len(minimized[0]) == 2:
        return det_3(minimized, kvector)

    det = 0
    sign = 1

    for min, k in zip(minimized, kvector):
        a, b = minimize_matrix(min, len(min))
        det += sign * k * det_3(a, b)
    return det
