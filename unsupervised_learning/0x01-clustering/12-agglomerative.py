#!/usr/bin/env python3
"""
module which contains agglomerative function
"""
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """
    that performs agglomerative clustering on a dataset:

    - X is a numpy.ndarray of shape (n, d) containing the dataset
    - dist is the maximum cophenetic distance for all clusters
    - Performs agglomerative clustering with Ward linkage
    - Displays the dendrogram with each cluster displayed in a different color
    Returns: clss, a numpy.ndarray of shape (n,)
             containing the cluster indices for each data point
    """
    # computing clusters
    linkage = scipy.cluster.hierarchy.linkage(X, method='ward')
    clss = scipy.cluster.hierarchy.fcluster(linkage, t=dist, criterion='distance')

    # plotting
    fig = plt.figure()
    dendogram = scipy.cluster.hierarchy.dendrogram(linkage, color_threshold=dist)
    plt.show()

    return clss
