#!/usr/bin/env python3
"""
module which contains Dataset class
"""
import tensorflow_datasets as tfds


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
