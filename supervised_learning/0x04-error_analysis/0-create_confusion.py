#!/usr/bin/env python3
"""
module which contains create_confusion_matrix function
"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    creates a confusion matrix:

    * labels is a one-hot numpy.ndarray of shape (m, classes)
      containing the correct labels for each data point
        - m is the number of data points
        - classes is the number of classes
    * logits is a one-hot numpy.ndarray of shape (m, classes)
      containing the predicted labels
    Returns: a confusion numpy.ndarray of shape (classes, classes) with row
             indices representing the correct labels and column indices
             representing the predicted labels
    """
    return labels.T @ logits
