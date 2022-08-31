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
    new = data.copy()
    for i, item in enumerate(new):
        if i == 0:
            new[i] = float(item)
        else:
            vt = (beta * new[i - 1]) + ((1 - beta) * item)
            new[i] = vt / 1 - (beta ** i)
    return new
