#!/usr/bin/env python3
"""
module which contains add_arrays cat_matrices2D
"""


def add_arrays(arr1, arr2):
    """
    returns matrix addition from two arrays without numpy
    """
    length = len(arr1)
    if length != len(arr2):
        return None
    add = []
    i = 0
    while i < length:
        add.append(arr1[i] + arr2[i])
        i += 1
    return add


def cat_matrices2D(mat1, mat2, axis=0):
    """
    cocatenates two 2D matrix
    """

    mat1copy = list(map(list, mat1))
    if not axis:
        len1 = len(mat1copy[0])
        len2 = len(mat2[0])
        if len1 == len2:
            for item in mat2:
                mat1copy.append(item)
            return mat1copy
    if axis == 1:
        len1 = len(mat1copy)
        len2 = len(mat2)
        if len1 == len2:
            i = 0
            for item in mat2:
                mat1copy[i] = mat1copy[i] + item
                i += 1
            return mat1copy
    return None


def cat_matrices(mat1, mat2, axis=0):
    """
    concatenates two matrices along a specific axis
    """
    aux1 = mat1
    aux2 = mat2
    i = 0
    while type(aux1) == list:
        if i == 0 and axis == 1 and type(aux1) is list and type(aux2) is list:
            if len(aux1) != len(aux2):
                return None
            i += 1
        aux1 = aux1[0]
        aux2 = aux2[0]
    if i == 1:
        return add_arrays(mat1, mat2)
    if i == 2:
        return cat_matrices2D(mat1, mat2, axis)
    if i == 4:
        matcopy1 = list(map(lambda x: list(map(list, x)), mat1))
        len1 = len(mat1)
        len2 = len(mat1[0])
        len3 = len(mat1[0][0])
        if axis == 1:
            for i in range(len1):
                for j in range(len2):
                    matcopy1[i] += mat2[i]
        if axis == 3:
            for i in range(len1):
                for j in range(len2):
                    for k in range(len3):
                        matcopy1[i][j][k] += mat2[i][j][k]
        return matcopy1
