#!/usr/bin/env python3
"""
module which contains the Poisson class
"""


class Poisson():
    """
    represents a poisson distribution
    """
    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """
        initializes poisson distribution
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
            self.lambtha = float(p / i)
        else:
            if lambtha > 0:
                self.lambtha = float(lambtha)
            else:
                raise ValueError('lambtha must be a positive value')

    def factorial(self, n):
        """
        calculates the factorial of n
        """
        if n == 0:
            return 1
        kFactorial = 1
        for i in range(2, n + 1):
            kFactorial *= i
        return kFactorial

    def pmf(self, k):
        """
        Calculates the value of the PMF for a given number of “successes”
        """
        if not isinstance(k, int):
            k = int(k)
        if k < 1:
            return 0
        num = (self.e ** -self.lambtha) * (self.lambtha ** k)
        return num / self.factorial(k)

    def cdf(self, k):
        """
        Calculates the value of the CDF for a given number of “successes”
        """
        sigma = 0
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0
        for i in range(0, k + 1):
            sigma += self.pmf(i)
        return sigma
