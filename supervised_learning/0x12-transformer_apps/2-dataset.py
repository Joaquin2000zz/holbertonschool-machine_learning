#!/usr/bin/env python3
"""
module which contains Dataset class
"""
import tensorflow_datasets as tfds
import tensorflow.compat.v2 as tf


class Dataset:
    """
    loads and preps a dataset for machine translation:
    """

    def __init__(self):
        """
        creates the instance attributes:
        - data_train: which contains the ted_hrlr_translate/pt_to_en
            tf.data.Dataset train split, loaded as_supervided
        - data_valid: which contains the ted_hrlr_translate/pt_to_en
            tf.data.Dataset validate split, loaded as_supervided
        - tokenizer_pt: is the Portuguese tokenizer
            created from the training set
        - tokenizer_en: is the English tokenizer created from the training set
        """
        pt2en = tfds.load(
            'ted_hrlr_translate/pt_to_en', as_supervised=True)
        self.data_train = pt2en['train']
        self.data_valid = pt2en['validation']

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            pt2en['train']
        )

        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

    def tokenize_dataset(self, data):
        """
        creates sub-word tokenizers for our dataset:
        - data: is a tf.data.Dataset whose examples are formatted
            as a tuple (pt, en)
        - pt: is the tf.Tensor containing the Portuguese sentence
        - en: is the tf.Tensor containing the corresponding English sentence
        - The maximum vocab size should be set to 2**15
        Returns: tokenizer_pt, tokenizer_en
          * tokenizer_pt: is the Portuguese tokenizer
          * tokenizer_en: is the English tokenizer
        """
        build = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus
        tokenizer_pt = build([pt.numpy() for pt, _ in data],
                             2 ** 15)
        tokenizer_en = build([en.numpy() for _, en in data],
                             2 ** 15)
        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """
        encodes a translation into tokens:
        - pt: is the tf.Tensor containing the Portuguese sentence
        - en: is the tf.Tensor containing the corresponding English sentence
        - The tokenized sentences should include the start
            and end of sentence tokens
        - The start token should be indexed as vocab_size
        - The end token should be indexed as vocab_size + 1
        Returns: pt_tokens, en_tokens
          * pt_tokens is a np.ndarray containing the Portuguese tokens
          * en_tokens is a np.ndarray. containing the English tokens
        """
        pt_vocab_size = self.tokenizer_pt.vocab_size
        en_vocab_size = self.tokenizer_en.vocab_size
        end = [pt_vocab_size + 1]
        middle = self.tokenizer_pt.encode(pt.numpy())
        pt_tokens = [pt_vocab_size] + middle + end
        end = [en_vocab_size + 1]
        middle = self.tokenizer_en.encode(en.numpy())
        en_tokens = [en_vocab_size] + middle + end
        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """
        acts as a tensorflow wrapper for the encode instance method
        - Make sure to set the shape of the pt and en return tensors
        - Update the class constructor def __init__(self):
        - update the data_train and data_validate attributes
          by tokenizing the examples
        """
        return tf.py_function(self.encode, (pt, en), (tf.int64, tf.int64))
