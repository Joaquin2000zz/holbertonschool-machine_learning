#!/usr/bin/env python3
"""
module which contains the Binomial class
"""


class Binomial():
    """
    represents a binomial distribution
    """

    def __init__(self, data=None, n=1, p=0.5):
        """
        initialize a binomial distribution
        """
        if data or data is not None:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            res = 0
            self.n = round(len(data) / 2)
            for value in data:
                if not isinstance(value, int) and not isinstance(value, float):
                    raise ValueError('list values must be integers or floats')
                res += value
            self.p = float(self.reduce(res / self.n))
        else:
            if n > 0:
                self.n = int(n)
            else:
                raise ValueError('n must be a positive value')
            if p < 0 or p > 1:
                raise ValueError('p must be greater than 0 and less than 1')
            self.p = float(p)

    def reduce(self, n):
        """
        function to reduce numbers to values
        between 0 and 1
        """
        if n < 0:
            return None
        if n <= 1:
            return float(n)
        ret = self.reduce(n / 10)

        return float(ret)
