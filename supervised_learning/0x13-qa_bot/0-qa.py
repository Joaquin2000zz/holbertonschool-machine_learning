#!/usr/bin/env python3
"""
module which contains question_answer function
"""
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


def question_answer(question, reference): 
    """
    finds a snippet of text within a reference document to answer a question:

    - question: is a string containing the question to answer
    - reference: is a string containing the reference document
                 from which to find the answer
    Returns: a string containing the answer
    - If no answer is found, return None
    - Your function should use the bert-uncased-tf2-qa model
      from the tensorflow-hub library
    - Your function should use the pre-trained BertTokenizer,
      bert-large-uncased-whole-word-masking-finetuned-squad,
      from the transformers library
    """
    # https://huggingface.co/models
    tokenizer = 'bert-large-uncased-whole-word-masking-finetuned-squad'
    tokenizer = BertTokenizer.from_pretrained(tokenizer)
    # https://tfhub.dev/
    model = hub.load('https://tfhub.dev/see--/bert-uncased-tf2-qa/1')   

    question_tokens = tokenizer.tokenize(question)

    reference_tokens = tokenizer.tokenize(reference)

    cls, sep = ['[CLS]'], ['[SEP]']
    tokens = cls + question_tokens + sep + reference_tokens + sep

    input_word_ids = tokenizer.convert_tokens_to_ids(tokens)

    input_mask = [1] * len(input_word_ids)

    len_question_t, len_reference_t = len(question_tokens), len(reference_tokens)
    input_type_ids = [0] * (1 + len_question_t + 1) + [1] * (len_reference_t + 1)

    input_word_ids, input_mask, input_type_ids = map(lambda t: tf.expand_dims(
        tf.convert_to_tensor(t, dtype=tf.int32), 0), (input_word_ids, input_mask, input_type_ids)
    )

    output = model([input_word_ids, input_mask, input_type_ids])
    
    # using `[1:]` will enforce an answer. `outputs[0][0][0]`
    # is the ignored '[CLS]' token logit
    short_start = tf.argmax(output[0][0][1:]) + 1
    short_end = tf.argmax(output[1][0][1:]) + 1
    answer_tokens = tokens[short_start: short_end + 1]
    answer = tokenizer.convert_tokens_to_string(answer_tokens)

    return answer
