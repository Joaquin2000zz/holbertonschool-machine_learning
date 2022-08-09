#!/usr/bin/env python3
"""
module which contains the Normal class
"""


from argparse import ONE_OR_MORE


class Normal():
    """
    represents an normal distribution
    """
    π = 3.1415926536
    e = 2.7182818285

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        initializes the normal object
        """
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
            self.stddev = self.heron(sigma / i)

        else:
            if stddev > 0:
                self.stddev = float(stddev)
            else:
                raise ValueError('stddev must be a positive value')
            self.mean = float(mean)

    def heron(self, p):
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

    def z_score(self, x):
        """
        Calculates the z-score of a given x-value
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        Calculates the x-value of a given z-score
        """
        return self.stddev * z + self.mean

    def pdf(self, x):
        """
        Calculates the value of the PDF for a given x-value
        """
        first = ((1 / (self.stddev * self.heron(self.π * 2))))
        second = self.e ** (- 0.5 * (((x - self.mean) / self.stddev) ** 2))
        return first * second

    def erf(self, x):
        """
        error function approximation
        """
        one = (x ** 3) / 3
        two = (x ** 5) / 10
        tree = (x ** 7) / 42
        four = (x ** 9) / 216
        return (2 / self.heron(self.π)) * (x - one + two - tree + four)

    def cdf(self, x):
        """
        Calculates the value of the CDF for a given x-value
        """
        res = self.erf((x - self.mean) / (self.stddev * self.heron(2)))
        return 0.5 * (1 + res)
