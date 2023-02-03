#!/usr/bin/env python3
"""
module which contains QA class and answer_loop function
"""
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer

class QA:
    """
    class which performs the answering
    """

    def __init__(self, reference):
        # https://huggingface.co/models
        tokenizer = 'bert-large-uncased-whole-word-masking-finetuned-squad'
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer)
        # https://tfhub.dev/
        self.model = hub.load('https://tfhub.dev/see--/bert-uncased-tf2-qa/1')
        self.reference = reference
        self.reference_tokens = self.tokenizer.tokenize(reference)

    def answer(self, question):
        """
        performs the answering
        """
        question_tokens = self.tokenizer.tokenize(question)

        cls, sep = ['[CLS]'], ['[SEP]']
        tokens = cls + question_tokens + sep + self.reference_tokens + sep
        
        input_word_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_word_ids)

        len_question_t = len(question_tokens)
        len_reference_t = len(self.reference_tokens)

        input_type_ids = [0] * (1 + len_question_t + 1) + [1] * (len_reference_t + 1)

        X = (input_word_ids, input_mask, input_type_ids)
        X = map(lambda t: tf.expand_dims(
            tf.convert_to_tensor(t, dtype=tf.int32), 0), X
            )
        output = self.model(list(X))

        # using `[1:]` will enforce an answer. `outputs[0][0][0]`
        # is the ignored '[CLS]' token logit
        short_start = tf.argmax(output[0][0][1:]) + 1
        short_end = tf.argmax(output[1][0][1:]) + 1
        answer_tokens = tokens[short_start: short_end + 1]
        answer = self.tokenizer.convert_tokens_to_string(answer_tokens)

        return answer

def answer_loop(reference):
    """
    answers questions from a reference text:

    - reference is the reference text
    - If the answer cannot be found in the reference text,
      respond with Sorry, I do not understand your question.
    """
    qa = QA(reference)
    stop = ['exit', 'quit', 'goodbye', 'bye']

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