#!/usr/bin/env python3
"""
module which contains the Normal class
"""


class Normal():
    """
    represents an normal distribution
    """
    e = 2.7182818285

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        initializes the normal object
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
            self.mean = float(p / i)
            sigma = 0
            for x in data:
                sigma += (x - self.mean) * (x - self.mean)
            self.stddev = self.HeronMethod(sigma / i)

        else:
            if stddev > 0:
                self.stddev = float(stddev)
            else:
                raise ValueError('stddev must be a positive value')
            self.mean = float(mean)

    def HeronMethod(self, p):
        """
        heron - method which calculates the srqt of a number
        @p: number to find their square root
        Return: the square root of p
        """
        x = p
        # seed
        xn = p / 2
        while 1:
            xn = 0.5 * (xn + (p / xn))
            if xn == x:
                break
            x = xn
        return float(x)

    def pdf(self, x):
        """
        Calculates the value of the PDF for a given time period
        """
        if x < 0:
            return 0
        return self.lambtha * self.e ** -self.lambtha * x

    def cdf(self, x):
        """
        Calculates the value of the CDF for a given time period
        """
        if x < 0:
            return 0
        return (1 - (self.e ** (-self.lambtha * x)))
