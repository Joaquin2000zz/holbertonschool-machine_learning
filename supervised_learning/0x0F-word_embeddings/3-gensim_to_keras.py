#!/usr/bin/env python3
"""
module which contains gensim_to_keras function
"""
import tensorflow as tf


def gensim_to_keras(model):
    """
    converts a gensim word2vec model to a keras Embedding layer:

    - model: is a trained gensim word2vec models
    Returns: the trainable keras Embedding
    """
    keyed_vectors = model.wv # structure holding the result of training
    weights = keyed_vectors.vectors # vectors themselves, a 2D numpy array

    embedding = tf.keras.layers.Embedding(
        input_dim=weights.shape[0],
        output_dim=weights.shape[1],
        weights=[weights]
    )
    return embedding
