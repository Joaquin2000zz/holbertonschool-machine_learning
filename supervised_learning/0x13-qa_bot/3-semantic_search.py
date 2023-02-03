#!/usr/bin/env python3
"""
module which contains question_answer function
"""
from glob import glob
import numpy as np
import tensorflow_hub as hub

def semantic_search(corpus_path, sentence): 
    """
    performs semantic search on a corpus of documents:

    - corpus_path: is the path to the corpus of reference documents
                   on which to perform semantic search
    - sentence: is the sentence from which to perform semantic search
    Returns: the reference text of the document most similar to sentence
    """
    model = hub.load('https://tfhub.dev/google/universal-sentence-encoder-large/5')
    X = [sentence]
    for ref in glob(f'{corpus_path}/*.md'):
        with open(ref, "r") as f:
            X.append(f.read())
    output = model(X)
    inner = np.inner(output, output)[0, 1:] # ignoring question
    idx = np.argmax(inner)
    return X[1 + idx]
