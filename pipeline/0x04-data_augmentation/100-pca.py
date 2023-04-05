#!/usr/bin/env python3
"""
module which contains pca_color function
"""
import numpy as np
import tensorflow as tf


def pca_color(image: tf.Tensor,
              alphas=np.random.normal(0, 0.1, 3)) -> tf.Tensor:
    """
    computes pca for images described in the AlexNet paper
    link: https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
    """
    # convert the image tensor to a 2D tensor
    h, w, c = image.shape
    flattened_image = tf.cast(tf.reshape(image, [-1, c]),
                              tf.float32)
    # compute z score for each datapoint
    mean = tf.math.reduce_mean(flattened_image, axis=0)
    centered_image = flattened_image - mean
    stdd = tf.math.reduce_std(centered_image, axis=0)
    centered_image /= stdd
    # compute sample covariance matrix
    covariance = tf.matmul(centered_image, centered_image,
                           transpose_a=True) / tf.cast(h * w, tf.float32) - 1
    # compute the eigenvectors and eigenvalues of the covariance matrix
    eigenvalues, eigenvectors = tf.linalg.eigh(covariance)

    alphas = alphas * eigenvalues
    projection_matrix = tf.matmul(eigenvectors, alphas[:, tf.newaxis])
      
    pca_image = centered_image + tf.transpose(projection_matrix)
    pca_image *= stdd
    pca_image += mean
    # comment this line if you want cool images haha
    pca_image = tf.math.maximum(tf.math.minimum(pca_image, 255)
                                , 0)
    pca_image = tf.cast(pca_image, dtype='uint8')
    pca_image = tf.reshape(pca_image, [h, w, c])

    return pca_image
