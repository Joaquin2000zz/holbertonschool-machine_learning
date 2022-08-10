#!/usr/bin/env python3
"""
module which contains the Binomial class
"""


class Binomial:
    """class Binomial that represents a binomial distribution"""

    def __init__(self, data=None, n=1, p=0.5):
        """Class contructor"""
        if data or data is not None:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            aux = 0
            i = 0
            aux2 = 0
            for item in data:
                aux += item
                i += 1
            m = aux / i
            for item2 in data:
                aux2 += (item2 - m) ** 2
            x = aux2 / i
            self.p = float(1 - (x / m))
            self.n = round(m / self.p)
            self.p = float(m / self.n)
        else:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = n
            self.p = p

    def factorial(self, x):
        """factorial function"""
        if x == 0:
            return 1
        return (self.factorial(x - 1) * x)

    def pmf(self, k):
        """Calculates the value of the PMF for a given number of successes"""
        if type(k) is not int:
            k = int(k)
        if k < 0 or k > self.n:
            return 0
        com = self.factorial(self.n) / (self.factorial
                                        (self.n - k) * self.factorial(k))
        return (com * (self.p ** k) * ((1 - self.p) ** (self.n - k)))
