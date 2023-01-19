#!/usr/bin/env python3
"""
module which contains  class
"""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention

class EncoderBlock(tf.keras.layers.Layer):
    """
    create an encoder block for a transformer:
    """

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        - dm: the dimensionality of the model
        - h: the number of heads
        - hidden: the number of hidden units in the fully connected layer
        - drop_rate: the dropout rate
        - Sets the following public instance attributes:
          * mha: a MultiHeadAttention layer
          * dense_hidden: the hidden dense layer with hidden units
              and relu activation
          * dense_output: the output dense layer with dm units
          * layernorm1: the first layer norm layer, with epsilon=1e-6
          * layernorm2: the second layer norm layer, with epsilon=1e-6
          * dropout1: the first dropout layer
          * dropout2: the second dropout layer
        """
        super(EncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(units=hidden,
                                                  activation='relu')
        self.dense_output = tf.keras.layers.Dense(units=dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """
        x: a tensor of shape (batch, input_seq_len, dm)
            containing the input to the encoder block
        training: a boolean to determine if the model is training
        mask: the mask to be applied for multi head attention
        Returns: a tensor of shape (batch, input_seq_len, dm)
            containing the block’s output
        """
        outputs, weights = self.mha(x, x, x, mask)

        outputs = self.dropout1(outputs, training)
        norm_outputs = self.layernorm1(x + outputs)
        hidden_outputs = self.dense_hidden(norm_outputs)
        forward_outputs = self.dense_output(hidden_outputs)
        forward_outputs = self.dropout2(forward_outputs, training)
        return self.layernorm2(norm_outputs + forward_outputs)
