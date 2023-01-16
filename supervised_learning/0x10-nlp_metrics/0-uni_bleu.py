#!/usr/bin/env python3
"""
module which contains ngram_bleu function
"""
import numpy as np


def n_gram(sentence, n=1):
    """
    generates ngram segmentation
    """
    ngram = []
    m = len(sentence)
    for i in range(m):
        new = ' '.join(sentence[i: i + n])
        if new not in ngram:
            ngram.append(new)
    return ngram


def uni_bleu(references, sentence):
    """
    calculates the n-gram BLEU score for a sentence:

    - references: is a list of reference translations
      * each reference translation is a list of the words in the translation
    - sentence: is a list containing the model proposed sentence
    - n: size of the n-gram to use for evaluation
    Returns: the n-gram BLEU score
    """
    count_clip = 0
    count = 0
    m = len(sentence)
    ngram = n_gram(sentence)
    gram_refs = [n_gram(ref) for ref in references]
    for word in ngram:
        count_clip += np.max([ref.count(word) for ref in gram_refs])
        count += ngram.count(word)

    r = len(references[np.argmin([abs(len(x) - m) for x in references])])
    if m > r:
        brevity_penalty = 1
    else:
        brevity_penalty = np.exp(1 - (r / m))
    return brevity_penalty * (count_clip / count)
