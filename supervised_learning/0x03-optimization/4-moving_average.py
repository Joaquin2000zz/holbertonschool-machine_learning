#!/usr/bin/env python3
"""
module which contains moving_average function
"""


def moving_average(data, beta):
    """
    Calculates the weighted moving average of a data set:

    data is the list of data to calculate the moving average of
    beta is the weight used for the moving average
    Your moving average calculation should use bias correction
    Returns: a list containing the moving averages of data
    """
    new = []
    vt = 0
    for i, alpha in enumerate(data):
        vt = (beta * vt) + ((1 - beta) * alpha)
        bias = (1 - (beta ** (i + 1)))
        new.append(vt / bias)
    return new
