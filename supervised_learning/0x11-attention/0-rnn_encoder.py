#!/usr/bin/env python3
"""
module which contains  function
"""
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """
    inherits from tensorflow.keras.layers.Layer
    to encode for machine translation
    """

    def __init__(self, vocab, embedding, units, batch):
        """
        - vocab: is an integer representing the size of the input vocabulary
        - embedding: is an integer representing the dimensionality
            of the embedding vector
        - units: is an integer representing the number
            of hidden units in the RNN cell
        - batch: is an integer representing the batch size
        - Sets the following public instance attributes:
          * batch - the batch size
          * units - the number of hidden units in the RNN cell
          * embedding - a keras Embedding layer that converts words
              from the vocabulary into an embedding vector
          * gru - a keras GRU layer with units units
        - Should return both the full sequence of outputs
            as well as the last hidden state
        - Recurrent weights should be initialized with glorot_uniform
        """
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(
            vocab,
            embedding,
            embeddings_initializer='uniform',
            embeddings_regularizer=None,
            activity_regularizer=None,
            embeddings_constraint=None,
            mask_zero=False,
            input_length=None,
        )
        self.gru = tf.keras.layers.GRU(units,
                                       activation='tanh',
                                       recurrent_activation='sigmoid',
                                       use_bias=True,
                                       kernel_initializer='glorot_uniform',
                                       recurrent_initializer='orthogonal',
                                       bias_initializer='zeros',
                                       kernel_regularizer=None,
                                       recurrent_regularizer=None,
                                       bias_regularizer=None,
                                       activity_regularizer=None,
                                       kernel_constraint=None,
                                       recurrent_constraint=None,
                                       bias_constraint=None,
                                       dropout=0.0,
                                       recurrent_dropout=0.0,
                                       return_sequences=True,
                                       return_state=True,
                                       go_backwards=False,
                                       stateful=False,
                                       unroll=False,
                                       time_major=False,
                                       reset_after=True,
                                       )

    def initialize_hidden_state(self):
        """
        Initializes the hidden states for the RNN cell to a tensor of zeros
        Returns: a tensor of shape (batch, units)
            containing the initialized hidden states
        """
        return tf.initializers.zeros()(shape=(self.batch, self.units))

    def call(self, x, initial):
        """
        - x: is a tensor of shape (batch, input_seq_len) containing the input
            to the encoder layer as word indices within the vocabulary
        - initial: is a tensor of shape (batch, units)
            containing the initial hidden state
        Returns: outputs, hidden
          - outputs: is a tensor of shape (batch, input_seq_len, units)
              containing the outputs of the encoder
          - hidden: is a tensor of shape (batch, units)
              containing the last hidden state of the encoder
        """
        outputs, state = self.gru(self.embedding(x), initial_state=initial)

        return outputs, state
