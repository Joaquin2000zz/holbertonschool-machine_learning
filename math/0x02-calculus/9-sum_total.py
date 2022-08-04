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
    flag = str(n)
    if flag.replace('.', '', 1).isnumeric() is False:
        return None
    if n == 0:
        return 0
    return summation_i_squared(n - 1) + n ** 2
