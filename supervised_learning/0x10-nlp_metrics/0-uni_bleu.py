#!/usr/bin/env python3
"""
module which contains uni_bleu function
"""
import numpy as np


def uni_bleu(references, sentence):
    """
    calculates the unigram BLEU score for a sentence:

    - references: is a list of reference translations
      * each reference translation is a list of the words in the translation
    - sentence: is a list containing the model proposed sentence
    Returns: the unigram BLEU score
    """
    count_clip = 0
    count = 0
    n = 0
    for word in sentence:
        count_clip += np.max([ref.count(word) for ref in references])
        count += sentence.count(word)
        n += 1

    r = len(references[np.argmin([abs(len(x) - n) for x in references])])
    if n > r:
        brevity_penalty = 1
    else:
        brevity_penalty = np.exp(1 - (r / n))
    return brevity_penalty * (count_clip / count)
