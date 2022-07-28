#!/usr/bin/env python3
"""
module which contains np_slice
"""


def np_slice(matrix, axes={}):
    """
    sclices untill 3D matrices
    """
    ret = matrix
    for key, value in axes.items():
        if key == 0:
            length = len(value)
            if length == 2:
                ret = ret[value[0]:value[1]]
            elif length == 3:
                ret = ret[value[0]:value[1]:value[2]]
            else:
                ret = ret[value[0]:]
        if key == 1:
            length = len(value)
            if length == 2:
                ret = ret[:, value[0]:value[1]]
            elif length == 3:
                ret = ret[:, value[0]:value[1]:value[2]]
            else:
                ret = ret[:, value[0]:]
        if key == 2:
            length = len(value)
            if length == 2:
                ret = ret[:, :, value[0]:value[1]]
            elif length == 3:
                ret = ret[:, :, value[0]:value[1]:value[2]]
            else:
                ret = ret[:, :, value[0]:]
    return ret
