#!/usr/bin/env python3
"""
module which contains  class
"""
import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    perform multi head attention:
    """

    def __init__(self, dm, h):
        """
        - dm: is an integer representing the dimensionality of the model
        - h: is an integer representing the number of heads
        - dm: is divisible by h
        - Sets the following public instance attributes:
          * h - the number of heads
          * dm - the dimensionality of the model
          * depth - the depth of each attention head
          * Wq - a Dense layer with dm units, used to generate
              the query matrix
          * Wk - a Dense layer with dm units, used to generate
              the key matrix
          * Wv - a Dense layer with dm units, used to generate
              the value matrix
          * linear - a Dense layer with dm units, used to generate
              the attention output
        """
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // h
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def call(self, Q, K, V, mask):
        """
        Q: is a tensor of shape (batch, seq_len_q, dk)
            containing the input to generate the query matrix
        K: is a tensor of shape (batch, seq_len_v, dk)
            containing the input to generate the key matrix
        V: is a tensor of shape (batch, seq_len_v, dv)
            containing the input to generate the value matrix
        mask: is always None
        Returns: output, weights
          * outputs: tensor with its last two dimensions as
              (..., seq_len_q, dm) containing the scaled dot product attention
          * weights: a tensor with its last three dimensions as
              (..., h, seq_len_q, seq_len_v) containing the attention weights
        """
        batch = tf.shape(Q)[0]
        # reshape data into desired shape
        Q = self.reshape_tensor(self.Wq(Q), batch)
        K = self.reshape_tensor(self.Wk(K), batch)
        V = self.reshape_tensor(self.Wv(V), batch)
        outputs, weights = sdp_attention(Q, K, V, mask)

        outputs = tf.transpose(outputs, perm=[0, 2, 1, 3])
        outputs = tf.reshape(outputs, (batch, -1, self.dm))

        return self.linear(outputs), weights

    def reshape_tensor(self, x, batch):
        """
        reshapes the linearity projected queries, keys, and values
        in such a manner as to allow the attention heads
        to be computed in parallel
        """
        x = tf.reshape(x, shape=(batch, -1, self.h, self.depth))
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        return x
