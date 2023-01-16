#!/usr/bin/env python3
"""
module which contains ngram_bleu function
"""
import numpy as np


def n_gram(sentence, n):
    """
    generates ngram segmentation
    """
    if n == 1:
        return sentence
    ngram = []
    m = len(sentence)
    for i in range(m - (n - 1)):
        new = ' '.join(sentence[i: i + n])
        if new not in ngram:
            ngram.append(new)
    return ngram


def ngram_bleu(references, sentence, n):
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
    ngram = n_gram(sentence, n)
    gram_refs = [n_gram(ref, n) for ref in references]
    for word in ngram:
        count_clip += np.max([ref.count(word) for ref in gram_refs])
        count += ngram.count(word)

    return count_clip / count

def cumulative_bleu(references, sentence, n):
    """
    calculates the cumulative n-gram BLEU score for a sentence:

    - references: is a list of reference translations
      * each reference translation is a list of the words in the translation
    - sentence: is a list containing the model proposed sentence
    - n: is the size of the largest n-gram to use for evaluation
    - All n-gram scores should be weighted evenly
    Returns: the cumulative n-gram BLEU score
    """
    m = len(sentence)

    cum_bleu = [ngram_bleu(references, sentence, i) for i in range(1, n + 1)]
    
    r = len(references[np.argmin([abs(len(x) - m) for x in references])])
    if m > r:
        brevity_penalty = 1
    else:
        brevity_penalty = np.exp(1 - (r / m))
    
    return brevity_penalty * np.sum(np.log(np.exp(cum_bleu)) / n)
