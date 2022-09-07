#!/usr/bin/env python3
"""
module which contains np_slice
"""

def sword(axes, i):
    """
    make slice object
    """
    axis = axes.get(i)

    if not axis:
        return slice(None)
    n = len(axis)
    if n == 1:
        return slice(axis[0])
    if n == 2:
        return slice(axis[0], axis[1])
    if n == 3:
        return slice(axis[0], axis[1], axis[2])

def rec(matrix, i, axes):
    """recursion needed
        @i: i dimention
        @matrix: matix to traverse and slice
    """
    ret = None
    print(f"dimension {i}")

    print(f"axes dict {axes}")



    if matrix.__class__.__name__ == "ndarray":
        if matrix[0].__class__.__name__ == "ndarray":
    
            print("entro de nuevo a la recursión?")
            
            ret = rec(matrix[0][sword(axes, i)].copy(), i + 1, axes)

            if isinstance(matrix[0][0], int):
                if not axes.get(i):
                    return matrix[:]

                cut = sword(axes, i)
                return matrix[cut]
    print(f"estoy saliendo por acá con matrix {matrix}")
    return ret


def np_slice(matrix, axes={}):
    """
    sclices untill 3D matrices
    """
    i = 0
    knife = (sword(axes, i),)
    aux = matrix
    ret = matrix[knife]
    while isinstance(aux[0], list):
        i +=1
        knife.append(sword(axes, i))
        aux = aux[0]
        ret[knife]

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
                ret = ret[:value[0]]
        if key == 1:
            length = len(value)
            if length == 2:
                ret = ret[:, value[0]:value[1]]
            elif length == 3:
                ret = ret[:, value[0]:value[1]:value[2]]
            else:
                ret = ret[:, :value[0]]
        if key == 2:
            length = len(value)
            if length == 2:
                ret = ret[:, :, value[0]:value[1]]
            elif length == 3:
                ret = ret[:, :, value[0]:value[1]:value[2]]
            else:
                ret = ret[:, :, :value[0]]
    """
    return ret
