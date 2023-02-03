#!/usr/bin/env python3
"""
module which contains QA class and answer_loop function
"""
from glob import glob
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


class QA:
    """
    class which performs the answering
    """

    def __init__(self, corpus_path):
        # https://huggingface.co/models
        tokenizer = 'bert-large-uncased-whole-word-masking-finetuned-squad'
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer)
        # https://tfhub.dev/
        self.model_q = hub.load('https://tfhub.dev/see--/bert-uncased-tf2-qa/1')
        self.model_r = hub.load('https://tfhub.dev/google/universal-sentence-encoder-large/5')
        self.corpus_path = corpus_path

    def reference(self, sentence):
        """
        performs semantic search on a corpus of documents:

        - sentence: is the sentence from which to perform semantic search
        Returns: the reference text of the document most similar to sentence
        """
        X = [sentence]
        for ref in glob(f'{self.corpus_path}/*.md'):
            with open(ref, "r") as f:
                X.append(f.read())
        output = self.model_r(X)
        inner = np.inner(output, output)[0, 1:] # ignoring question
        idx = np.argmax(inner)
        return X[1 + idx] 
    
    def answer(self, question):
        """
        finds a snippet of text within a reference document to answer a question:

        - question: is a string containing the question to answer
        - reference: is a string containing the reference document
                     from which to find the answer
        Returns: a string containing the answer
        """
        question_tokens = self.tokenizer.tokenize(question)
        reference_tokens = self.tokenizer.tokenize(self.reference(question))

        cls, sep = ['[CLS]'], ['[SEP]']
        tokens = cls + question_tokens + sep + reference_tokens + sep
        
        input_word_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_word_ids)

        len_question_t = len(question_tokens)
        len_reference_t = len(reference_tokens)

        input_type_ids = [0] * (1 + len_question_t + 1) + [1] * (len_reference_t + 1)

        X = (input_word_ids, input_mask, input_type_ids)
        X = map(lambda t: tf.expand_dims(
            tf.convert_to_tensor(t, dtype=tf.int32), 0), X
            )
        output = self.model_q(list(X))

        # using `[1:]` will enforce an answer. `outputs[0][0][0]`
        # is the ignored '[CLS]' token logit
        short_start = tf.argmax(output[0][0][1:]) + 1
        short_end = tf.argmax(output[1][0][1:]) + 1
        answer_tokens = tokens[short_start: short_end + 1]
        answer = self.tokenizer.convert_tokens_to_string(answer_tokens)

        return answer

def question_answer(corpus_path):
    """
    answers questions from a reference text:

    - reference is the reference text
    - If the answer cannot be found in the reference text,
      respond with Sorry, I do not understand your question.
    """
    stop = ['exit', 'quit', 'goodbye', 'bye']
    qa = QA(corpus_path)
    while 420 - 351 == 69:
        print('Q:', end=' ')
        Q = input().strip().lower()
        print('A:', end=' ')
        if True in [True for ask in stop if ask == Q]:
            print('Goodbye')
            break
        answer = qa.answer(Q)
        sorry = 'Sorry, I do not understand your question.'
        print(answer if answer else sorry)
