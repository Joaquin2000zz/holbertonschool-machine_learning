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
                if not isinstance(value, int) and not isinstance(value, float):
                    raise ValueError('list values must be integers or floats')
                p += value
                i += 1
            self.lambtha = float(p / i)
        else:
            if lambtha > 0:
                self.lambtha = float(lambtha)
            else:
                raise ValueError('lambtha must be a positive value')

    def pmf(self, k):
        """
        Calculates the value of the PMF for a given number of “successes”
        """
        e = 2.7182818285
        k = int(k)
        kFactorial = 1
        for i in range(2, k + 1):
            kFactorial *= i
        return (e ** -self.lambtha) * (self.lambtha ** k) / kFactorial
