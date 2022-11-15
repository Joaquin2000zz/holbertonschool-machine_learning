#!/usr/bin/env python3
"""
module which contains optimum_k function
"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    tests for the optimum number of clusters by variance:

    - X is a numpy.ndarray of shape (n, d) containing the data set
    - kmin is a positive integer containing the minimum number
      of clusters to check for (inclusive)
    - kmax is a positive integer containing the maximum number
      of clusters to check for (inclusive)
    - iterations is a positive integer containing the maximum number
      of iterations for K-means
    - This function should analyze at least 2 different cluster sizes

    - You may use at most 2 loops
    Returns: results, d_vars, or None, None on failure
        - results is a list containing the outputs of K-means
          for each cluster size
        - d_vars is a list containing the difference in variance from
          the smallest cluster size for each cluster size
    """
    try:
        if not isinstance(X, np.ndarray) or len(X.shape) != 2:
            return None, None
        if not isinstance(kmin, int) or kmin < 1 or kmin >= X.shape[0]:
            return None, None
        if isinstance(kmax, int):
            if kmax < kmin or kmax > X.shape[0]:
                return None, None
        elif not kmax:
            kmax = X.shape[0]
        else:
            return None, None
        if not isinstance(iterations, int) or iterations < 1:
            return None, None

        results = []
        d_vars = []
        for k in range(kmin, kmax + 1):
            # obtaining centroids means for each cluster
            # and clss that contains the indexes of which
            # cluster each data point belongs to
            C, clss = kmeans(X, k, iterations)
            results.append((C, clss))
            # computing the variance which says
            # how disperse the datapoints are
            d_var = variance(X, C)
            # obtaining first variance to compute
            # the difference between k1 var and kth
            if k == kmin:
                min_var = d_var
            d_vars.append(min_var - d_var)
        return results, d_vars
    except Exception:
        return None, None
