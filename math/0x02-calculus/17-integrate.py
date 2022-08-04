#!/usr/bin/env python3
"""
module which contains poly_integral
"""


def poly_integral(poly, C=0):
    """
    calculates the integral of a polynomial
    is a list of coefficients representing a polynomial
    the index of the list represents the power of x that the coefficient
    belongs to
    Example: if f(x) = x^3 + 3x +5, poly is equal to [5, 3, 0, 1]
    C is an integer representing the integration constant
    If a coefficient is a whole number, it should be represented as an integer
    If poly or C are not valid, return None
    Return a new list of coefficients representing the integral of
    the polynomial
    The returned list should be as small as possible
    """
    i = 1
    if (not poly or type(poly) != list or
        (type(C) != int and type(C) != float)):
        return None
    ret = [C]
    for mono in poly:
        if mono != 0:
            if int(mono / i) != float(mono / i):
                ret.append(mono / i)
            else:
                ret.append(int(mono / i))
        else:
            ret.append(mono)
        i += 1
    return ret
