#!/usr/bin/env python3
"""
contains mat_mul function
"""


def mat_mul(mat1, mat2):
    """
    returns matrix multiplication from two arrays without numpy
    """
    height = len(mat2)
    width = len(mat1[0])
    if height != width:
        return None
    ret = []
    newwidth = len(mat2[0])
    newheight = len(mat1)

    for i in range(newheight):
        child = []
        for j in range(newwidth):
            res = 0
            for k in range(height):
                res += (mat1[i][k] * mat2[k][j])
            child.append(res)
        ret.append(child)
    return ret
