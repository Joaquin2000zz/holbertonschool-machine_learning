#!/usr/bin/env python3
"""
module which contains the Poisson class
"""


class Poisson():
    """
    represents a poisson distribution
    """
    def __init__(self, data=None, lambtha=1.):
        """
        initializes poisson distribution
        """
        p = None
        if data:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            i = 0
            p = 0
            for value in data:
                p += value
                i += 1
            self.lambtha = p / i
        else:
            if lambtha > 0:
                self.lambtha = float(lambtha)
            else:
                raise ValueError('lambtha must be a positive value')
