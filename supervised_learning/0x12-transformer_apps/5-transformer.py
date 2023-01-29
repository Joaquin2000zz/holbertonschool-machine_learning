#!/usr/bin/env python3
"""
module which contains all functions and classes
which performs the transformer model
"""
import numpy as np
import tensorflow as tf


def positional_encoding(max_seq_len, dm):
    """
    calculates the positional encoding for a transformer:

    - max_seq_len: is an integer representing the maximum sequence length
    - dm: is the model depth
    Returns: a numpy.ndarray of shape (max_seq_len, dm)
        containing the positional encoding vectors
    """
    P = np.zeros((max_seq_len, dm))
    for k in range(max_seq_len):
        for i in np.arange(int(dm / 2)):
            denominator = np.power(10000, 2 * i / dm)
            P[k, 2 * i] = np.sin(k / denominator)
            P[k, 2 * i + 1] = np.cos(k / denominator)
    return P

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
    if mask is not None:
        mask *= tf.cast(-1e9, dtype=tf.float32)
        scaled += mask
    weights = tf.nn.softmax(scaled, axis=-1)
    outputs = tf.matmul(weights, V)

    return outputs, weights


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

    def reshape_tensor(self, x, batch):
        """
        reshapes the linearity projected queries, keys, and values
        in such a manner as to allow the attention heads
        to be computed in parallel
        """
        x = tf.reshape(x, (batch, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])


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
        batch_size = tf.shape(Q)[0]
        Q = self.Wq(Q)
        K = self.Wk(K)
        V = self.Wv(V)
        Q = self.reshape_tensor(Q, batch_size)
        K = self.reshape_tensor(K, batch_size)
        V = self.reshape_tensor(V, batch_size)
        output, weights = sdp_attention(Q, K, V, mask)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(output,
                                      (batch_size, -1, self.dm))
        output = self.linear(concat_attention)
        return output, weights

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

class DecoderBlock(tf.keras.layers.Layer):
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
            * mha1: the first MultiHeadAttention layer
            * mha2: the second MultiHeadAttention layer
            * dense_hidden: the hidden dense layer with hidden units
                and relu activation
            * dense_output: the output dense layer with dm units
            * layernorm1: the first layer norm layer, with epsilon=1e-6
            * layernorm2: the second layer norm layer, with epsilon=1e-6
            * layernorm3: the third layer norm layer, with epsilon=1e-6
            * dropout1: the first dropout layer
            * dropout2: the second dropout layer
            * dropout3: the third dropout layer
        """
        super(DecoderBlock, self).__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden,
                                                  activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training,
             look_ahead_mask, padding_mask):
        """
        x: a tensor of shape (batch, target_seq_len, dm) containing
            the input to the decoder block
        encoder_output: a tensor of shape (batch, input_seq_len, dm)
            containing the output of the encoder
        training: a boolean to determine if the model is training
        look_ahead_mask: the mask to be applied to the first
            multi head attention layer
        padding_mask: the mask to be applied to the second
            multi head attention layer
        Returns: a tensor of shape (batch, target_seq_len, dm)
            containing the block’s output
        """
        # multihead of embedding output
        o_mha1, w_mha1 = self.mha1(x, x, x, look_ahead_mask)
        o_mha1 = self.dropout1(o_mha1, training)
        norm1 = self.layernorm1(x + o_mha1)
        # multihead of encoder output
        o_mha2, w_mha2 = self.mha2(norm1, encoder_output, encoder_output,
                                   padding_mask)
        o_mha2 = self.dropout2(o_mha2, training)
        norm2 = self.layernorm2(o_mha2 + norm1)
        # last linear layers
        hidden_outputs = self.dense_hidden(norm2)
        forward_outputs = self.dense_output(hidden_outputs)
        forward_outputs = self.dropout3(forward_outputs, training)
        norm3 = self.layernorm3(norm2 + forward_outputs)

        return norm3

class Encoder(tf.keras.layers.Layer):
    """
    create the encoder for a transformer:
    """

    def __init__(self, N, dm, h, hidden, input_vocab, max_seq_len,
                 drop_rate=0.1):
        """
        - N: the number of blocks in the encoder
        - dm: the dimensionality of the model
        - h: the number of heads
        - hidden: the number of hidden units in the fully connected layer
        - input_vocab: the size of the input vocabulary
        - max_seq_len: the maximum sequence length possible
        - drop_rate: the dropout rate
        - Sets the following public instance attributes:
          * N: the number of blocks in the encoder
          * dm: the dimensionality of the model
          * embedding: the embedding layer for the inputs
          * positional_encoding: a numpy.ndarray of shape (max_seq_len, dm)
          containing the positional encodings
          * blocks: a list of length N containing all of the EncoderBlock‘s
          * dropout: the dropout layer, to be applied
              to the positional encodings
        """
        super(Encoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [EncoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """
        x: a tensor of shape (batch, input_seq_len, dm)
            containing the input to the encoder
        training: a boolean to determine if the model is training
        mask: the mask to be applied for multi head attention
        Returns: a tensor of shape (batch, input_seq_len, dm)
            containing the encoder output
        """

        seq_length = x.shape[1]
        embedded_words = self.embedding(x)
        embedded_words *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        embedded_words += self.positional_encoding[:seq_length]

        x = self.dropout(embedded_words, training)

        for layer in self.blocks:
            x = layer(x, training, mask)
        return x

class Decoder(tf.keras.layers.Layer):
    """
    create the decoder for a transformer:
    """

    def __init__(self, N, dm, h, hidden, target_vocab, max_seq_len,
                 drop_rate=0.1):
        """
        - N: the number of blocks in the encoder
        - dm: the dimensionality of the model
        - h: the number of heads
        - hidden: the number of hidden units in the fully connected layer
        - target_vocab: the size of the target vocabulary
        - max_seq_len: the maximum sequence length possible
        - drop_rate: the dropout rate
        Sets the following public instance attributes:
          * N: the number of blocks in the encoder
          * dm: the dimensionality of the model
          * embedding: the embedding layer for the targets
          * positional_encoding: a numpy.ndarray of shape (max_seq_len, dm)
              containing the positional encodings
          * blocks: a list of length N containing all of the DecoderBlock‘s
          * dropout: the dropout layer, to be applied to the
              positional encodings
        """
        super(Decoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [DecoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask,
             padding_mask):
        """
        - x: a tensor of shape (batch, target_seq_len, dm)
            containing the input to the decoder
        - encoder_output: a tensor of shape (batch, input_seq_len, dm)
            containing the output of the encoder
        - training: a boolean to determine if the model is training
        - look_ahead_mask: the mask to be applied to the first
            multi head attention layer
        - padding_mask: the mask to be applied to the second
            multi head attention layer
        - Returns: a tensor of shape (batch, target_seq_len, dm)
            containing the decoder output
        """
        seq_length = x.shape[1]
        embedded_words = self.embedding(x)
        embedded_words *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        embedded_words += self.positional_encoding[:seq_length]

        x = self.dropout(embedded_words, training)

        for layer in self.blocks:
            x = layer(x, encoder_output, training,
                      look_ahead_mask, padding_mask)
        return x

class Transformer(tf.keras.Model):
    """
    create a transformer network:
    """

    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab,
                 max_seq_input, max_seq_target, drop_rate=0.1):
        """
        - N: the number of blocks in the encoder and decoder
        - dm: the dimensionality of the model
        - h: the number of heads
        - hidden: the number of hidden units in the fully connected layers
        - input_vocab: the size of the input vocabulary
        - target_vocab: the size of the target vocabulary
        - max_seq_input: the maximum sequence length possible for the input
        - max_seq_target: the maximum sequence length possible for the target
        - drop_rate - the dropout rate
        - Sets the following public instance attributes:
          * encoder: the encoder layer
          * decoder: the decoder layer
          * linear: a final Dense layer with target_vocab units
        """
        super(Transformer, self).__init__()
        self.encoder = Encoder(N, dm, h, hidden, input_vocab,
                               max_seq_input, drop_rate)
        self.decoder = Decoder(N, dm, h, hidden, target_vocab,
                               max_seq_target, drop_rate)
        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(self, inputs, target, training, encoder_mask,
             look_ahead_mask, decoder_mask):
        """
        - inputs: a tensor of shape (batch, input_seq_len)
            containing the inputs
        - target: a tensor of shape (batch, target_seq_len)
            containing the target
        - training: a boolean to determine if the model is training
        - encoder_mask: the padding mask to be applied to the encoder
        - look_ahead_mask: the look ahead mask to be applied to the decoder
        - decoder_mask: the padding mask to be applied to the decoder
        Returns: a tensor of shape (batch, target_seq_len, target_vocab)
            containing the transformer output
        """
        y = self.encoder(inputs, training, encoder_mask)
        y = self.decoder(target, y, training, look_ahead_mask, decoder_mask)
        return self.linear(y)
