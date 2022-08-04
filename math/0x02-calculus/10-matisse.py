#!/usr/bin/env python3
"""
module which contains poly_derivative
"""


def poly_derivative(poly):
    """
    poly is a list of coefficients representing a polynomial
    the index of the list represents the power of x that the coefficient
    belongs to
    Example: if f(x) = x^3 + 3x + 5, poly is equal to [5, 3, 0, 1]
    """
    i = 0
    ret = []
    if not poly or type(poly) != list:
        return None

    for mono in poly:
        if i > 0:
            ret.append(i * mono)
        i += 1
    res = all(ele == 0 for ele in ret)
    if res:
        return [0]
    return ret
