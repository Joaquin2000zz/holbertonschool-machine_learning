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
            i = 0
            for value in data:
                if not isinstance(value, int) and not isinstance(value, float):
                    raise ValueError('list values must be integers or floats')
                res += value
                i += 1

            mu = res / i
            variation = 0

            for x in data:
                variation += (x - mu) ** 2
            variation /= i
            self.p = 1 - (variation / mu)
            self.n = round(mu / self.p)
            self.p = mu / self.n
        else:
            if n > 0:
                self.n = int(n)
            else:
                raise ValueError('n must be a positive value')
            if p <= 0 or p >= 1:
                raise ValueError('p must be greater than 0 and less than 1')
            self.p = float(p)

    def pmf(self, k):
        """
        Calculates the value of the PMF for a given number of “successes”
        """
        if k <= 0 or k <= self.n:
            return 0
        nFact = self.factorial(self.n)
        comb = nFact / (self.factorial(self.n - k) * self.factorial(k))
        return comb * ((self.p ** k) * (1 - self.p) ** (self.n - k))

    def cdf(self, k):
        """
        Calculates the value of the CDF for a given number of successes
        """
        k = int(k)
        if k < 0 or k > self.n:
            return 0
        sigma = 0
        for i in range(k + 1):
            sigma += self.pmf(i)
        return sigma

    def factorial(self, n):
        """
        calculates the factorial of n
        """
        k = int(k)
        if n == 0:
            return 1
        kFactorial = 1
        for i in range(2, n + 1):
            kFactorial *= i
        return kFactorial
