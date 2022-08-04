#!/usr/bin/env python3
"""
contains and summation_i_squared function
"""


def summation_i_squared(n):
    """
    sigma notation with 1=0 to n with i^2
    returns none if values aren't numeric
    recursion needed to don't use any loop
    as the task says
    """
    if not n or type(n) != int or n < 1:
        return None
    return int((n * ((n + 1) * (2 * n + 1))) / 6)
