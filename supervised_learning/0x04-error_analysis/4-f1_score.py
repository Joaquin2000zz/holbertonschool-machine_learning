#!/usr/bin/env python3
"""
module which contains f1_score function
"""
import numpy as np
precision = __import__('2-precision').precision
sensitivity = __import__('1-sensitivity').sensitivity


def f1_score(confusion):
    """
    calculates the f1_score for each class in a confusion matrix:
    (2((precision * sensitivity) / (precision + sensitivity)))
    * confusion is a confusion numpy.ndarray of shape (classes, classes)
      where row indices represent the correct labels and column indices
      represent the predicted labels
        - classes is the number of classes
    Returns: a numpy.ndarray of shape (classes,)
             containing the f1_score of each class
    """
    P = precision(confusion)
    S = sensitivity(confusion)
    return 2 * (P * S / (P + S))
