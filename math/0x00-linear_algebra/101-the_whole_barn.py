#!/usr/bin/env python3
"""
module which contains add_matrices
"""


def rec(mat1, mat2):
    """
    recursion needed to the addition
    """
    if type(mat1) is list and type(mat2) is list:
        if type(mat1[0][0]) is int and type(mat2[0][0]) is int:
            app = []
            for k in range(len(mat1)):
                app.append([i + j for i, j in zip(mat1[k], mat2[k])])
            return app
        ret = rec(mat1[0], mat2[0])
        return ret


def add_matrices(mat1, mat2):
    """
    adds until 3D matrices
    """
    aux1 = mat1
    aux2 = mat2
    i = 0
    while type(aux1) != int:
        if type(aux1) is list and type(aux2) is list:
            if len(aux1) != len(aux2):
                return None
            aux1 = aux1[0]
            aux2 = aux2[0]
            i += 1
    if i == 1:
        return [i + j for i, j in zip(mat1, mat2)]

    if i == 2:
        ret = rec(mat1, mat2)
    if i == 4:
        ret0 = []
        ret1 = []
        wtf = {}
        for j in range(len(mat1[0])):
            wtf[j] = rec(mat1[0][j], mat2[0][j])
            if wtf.get(j):
                wtf[j].append(rec(mat1[1][j], mat2[1][j]))
        for i in range(len(wtf.keys())):
            ret0.append([wtf[i][0], wtf[i][1]])
            ret1.append(wtf[i][2])
        ret = [ret0, ret1]

    return ret
