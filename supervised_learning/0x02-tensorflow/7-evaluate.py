#!/usr/bin/env python3
"""
module which contains evaluate function
"""
import tensorflow.compat.v1 as tf


def evaluate(X, Y, save_path):
    """
    evaluates the output of a neural network:

    * X is a numpy.ndarray containing the input data to evaluate
    * Y is a numpy.ndarray containing the one-hot labels for X
    * save_path is the location to load the model from
    * You are not allowed to use tf.saved_model
    Returns: the network’s prediction, accuracy, and loss, respectively
    """
    with tf.Session() as sess:

        # this is used to open the metadata of a training
        saver = tf.train.import_meta_graph('{}.meta'.format(save_path))
        saver.restore(sess, '{}'.format(save_path))
        graph = tf.get_default_graph()       
        
        # obtaining y_pred, x, y, loss and accuracy
        y_pred = graph.get_collection('y_pred')[0]
        x = graph.get_collection('x')[0]
        y = graph.get_collection('y')[0]
        accuracy = graph.get_collection('accuracy')[0]
        loss = graph.get_collection('loss')[0]
        
        
        # training session
        prediction = sess.run(y_pred, feed_dict={x: X, y: Y})
        precision = sess.run(accuracy, feed_dict={x: X, y: Y})
        cost = sess.run(loss, feed_dict={x: X, y: Y})

        return prediction, precision, cost
