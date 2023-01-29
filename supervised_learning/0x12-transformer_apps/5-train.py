#!/usr/bin/env python3
"""
module which contains train_transformer function
"""
import tensorflow as tf

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super().__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    step = tf.cast(step, dtype=tf.float32)
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):
    """
    creates and trains a transformer model for machine translation of
    Portuguese to English using our previously created dataset:

    - N: the number of blocks in the encoder and decoder
    - dm: the dimensionality of the model
    - h: the number of heads
    - hidden: the number of hidden units in the fully connected layers
    - max_len: the maximum number of tokens per sequence
    - batch_size: the batch size for training
    - epochs: the number of epochs to train for
    - You should use the following imports:
    - Your model should be trained with Adam optimization with
        beta_1=0.9, beta_2=0.98, epsilon=1e-9
    - The learning rate should be scheduled using the following equation with
        warmup_steps=4000:

    - Returns the trained model
    """
    learning_rate = CustomSchedule(dm)

    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9,
                                         beta_2=0.98, epsilon=1e-9)
    compute_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    data = Dataset(batch_size, max_len)

    def loss_func(x, y):
        """
        computes loss
        """
        mask = tf.math.logical_not(tf.math.equal(x, 0))
        l0ss = compute_loss(x, y)
        mask = tf.cast(mask, dtype=l0ss.dtype)
        l0ss *= mask
        return tf.reduce_mean(l0ss)

    input_vocab = data.tokenizer_pt.vocab_size + 2
    target_vocab = data.tokenizer_en.vocab_size + 2

    transformer = Transformer(N, dm, h, hidden, input_vocab, target_vocab,
                              max_len, max_len)
    
    loss = tf.keras.metrics.Mean()
    accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    for epoch in range(epochs):
        loss.reset_states()
        accuracy.reset_states()

        for (batch, (x, y)) in enumerate(data.data_train):
            y_inp, y_real = y[:, :-1], y[:, 1:]

            encoder_mask, combined_mask, decoder_mask = create_masks(x, y_inp)

            with tf.GradientTape() as tape:
                yhat = transformer(x, y_inp, True, encoder_mask,
                                   combined_mask, decoder_mask)
                l0ss = loss_func(y_real, yhat)
            gradients = tape.gradient(l0ss, transformer.trainable_variables)
            optimizer.apply_gradients(zip(gradients,
                                      transformer.trainable_variables))
            loss(l0ss)
            accuracy(y_real, yhat)

            if batch % 50 == 0:
              print(f"Epoch {epoch + 1}, batch {batch}: loss {loss.result()} \
accuracy {accuracy.result()}")
        print(f"Epoch {epoch + 1}: loss {loss.result()} \
accuracy {accuracy.result()}")

    return transformer
