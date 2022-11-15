#!/usr/bin/env python3
"""
module which contains kmeans function
"""
import numpy as np
initialize = __import__('0-initialize').initialize


def kmeans(X, k, iterations=1000):
    """
    that performs K-means on a dataset:

    - X is a numpy.ndarray of shape (n, d) containing the dataset
      * n is the number of data points
      * d is the number of dimensions for each data point
    - k is a positive integer containing the number of clusters
    - iterations is a positive integer containing the maximum
      number of iterations that should be performed
    - If no change in the cluster centroids occurs between iterations,
      your function should return
    - Initialize the cluster centroids using a multivariate
      uniform distribution (based on0-initialize.py)
    - If a cluster contains no data points during the update step,
      reinitialize its centroid
    - You should use numpy.random.uniform exactly twice
    - You may use at most 2 loops
    Returns: C, clss, or None, None on failure
      - C is a numpy.ndarray of shape (k, d)
        containing the centroid means for each cluster
      - clss is a numpy.ndarray of shape (n,) containing the index
        of the cluster in C that each data point belongs to
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k < 1:
        return None, None
    if not isinstance(iterations, int) or iterations < 1:
        return None, None
    C = initialize(X, k)
    prev = None
    for _ in range(iterations):
        # computing euclidean distances from each point and the centroids
        distances = np.linalg.norm(X - C[:, np.newaxis], axis=-1)
        prev = C.copy()

        clss = distances.argmin(axis=0)
        for i in range(k):
            # Updating Centroids by taking mean of Cluster it belongs to
            if X[clss==i].size == 0:
                C = initialize(X, k)
            else:
                C[i] = X[clss==i].mean(axis=0) 
          
        distances = np.linalg.norm(X - C[:, np.newaxis], axis=-1)
        clss = np.argmin(distances, axis=0)
        if np.all(C == prev):
            break
    return C, clss
