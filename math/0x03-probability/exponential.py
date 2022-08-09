#!/usr/bin/env python3
"""
module which contains the Exponential class
"""


class Exponential():
    """
    represents an exponential distribution
    """
    e = 2.7182818285
    def __init__(self, data=None, lambtha=1.):
        """
        initializes the exponential object
        """
        p = None
        if data or data is not None:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            i = 0
            p = 0
            for value in data:
                if not isinstance(value, int) and not isinstance(value, float):
                    raise ValueError('list values must be integers or floats')
                p += value
                i += 1
            self.lambtha = float(1 / (p / i))
        else:
            if lambtha > 0:
                self.lambtha = float(lambtha)
            else:
                raise ValueError('lambtha must be a positive value')
