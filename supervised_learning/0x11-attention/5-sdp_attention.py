#!/usr/bin/env python3
"""
module which contains sdp_attention function
"""
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """
    calculates the scaled dot product attention:

    - Q: is a tensor with its last two dimensions as (..., seq_len_q, dk)
        containing the query matrix
    - K: is a tensor with its last two dimensions as (..., seq_len_v, dk)
        containing the key matrix
    - V: is a tensor with its last two dimensions as (..., seq_len_v, dv)
        containing the value matrix
    - mask: is a tensor that can be broadcast into (..., seq_len_q, seq_len_v)
        containing the optional mask, or defaulted to None
    - if mask is not None, multiply -1e9 to the mask and add
        it to the scaled matrix multiplication
    - The preceding dimensions of Q, K, and V are the same
    Returns: output, weights
      * outputs: tensor with its last two dimensions as
          (..., seq_len_q, dv) containing the scaled dot product attention
      * weights: a tensor with its last two dimensions as
          (..., seq_len_q, seq_len_v) containing the attention weights
    """
    dk = Q.shape[-1]
    QKT = tf.matmul(Q, K, transpose_b=True)
    scaling_factor = tf.sqrt(tf.cast(dk, dtype=tf.float32))
    scaled = QKT / scaling_factor
    if mask:
        scaled += tf.cast(-1e9, dtype=tf.float32) * mask
    weights = tf.nn.softmax(scaled, axis=-1)
    outputs = tf.matmul(weights, V)

    return outputs, weights
