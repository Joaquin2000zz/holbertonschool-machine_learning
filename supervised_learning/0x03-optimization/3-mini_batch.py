#!/usr/bin/env python3
"""
module which contains shuffle_data function
"""
import numpy as np
import tensorflow.compat.v1 as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid,
                     batch_size=32, epochs=5,
                     load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """
    trains a loaded neural network model using mini-batch gradient descent:

    * X_train is a numpy.ndarray of shape (m, 784)
      containing the training data
        - m is the number of data points
        - 784 is the number of input features
    * Y_train is a one-hot numpy.ndarray of shape (m, 10)
      containing the training labels
        - 10 is the number of classes the model should classify
    * X_valid is a numpy.ndarray of shape (m, 784)
      containing the validation data
    * Y_valid is a one-hot numpy.ndarray of shape (m, 10)
      containing the validation labels
    * batch_size is the number of data points in a batch
    * epochs is the number of times the training should
      pass through the whole dataset
    * load_path is the path from which to load the model
    * save_path is the path to where the model should
      be saved after training
    * Returns: the path where the model was saved
    * Your training function should allow for a smaller final batch
      (a.k.a. use the entire training set)
    * 1) import meta graph and restore session
    * 2) Get the following tensors and ops from the collection restored
        - x is a placeholder for the input data
        - y is a placeholder for the labels
        - accuracy is an op to calculate the accuracy of the model
        - loss is an op to calculate the cost of the model
        - train_op is an op to perform one pass of
          gradient descent on the model
    * 3) loop over epochs:
        - shuffle data
        - loop over the batches:
            + get X_batch and Y_batch from data
            + train your model
    * 4) Save session
    * You should use shuffle_data = __import__('2-shuffle_data').shuffle_data
    """
    with tf.Session() as session:
        # instance of tf.train.Saver() to save
        saver = tf.train.Saver()
        # this is used to open the metadata of a training
        load = tf.train.import_meta_graph('{}.meta'.format(load_path))
        load.restore(session, '{}'.format(load_path))
        graph = tf.get_default_graph()

        # obtaining y_pred, x, y, loss and accuracy
        x = graph.get_collection('x')[0]
        y = graph.get_collection('y')[0]
        accuracy = graph.get_collection('accuracy')[0]
        loss = graph.get_collection('loss')[0]
        train_op = graph.get_collection('train_op')[0]

        l = int(X_train.shape[1] / batch_size)
        for i in range(epochs):
            j = 0
            for k in range(l):
                X_t_batch = X_train[j: j + batch_size + 1]
                Y_t_batch = Y_train[j: j + batch_size + 1]
                X_t_batch, Y_t_batch = shuffle_data(X_t_batch, Y_t_batch)
                X_v_batch = X_train[j: j + batch_size + 1]
                Y_v_batch = Y_train[j: j + batch_size + 1]
                X_v_batch, Y_v_batch = shuffle_data(X_v_batch, Y_v_batch)

                # training session
                t_precision = session.run(accuracy, feed_dict={x: X_t_batch,
                                                               y: Y_t_batch})
                t_cost = session.run(loss, feed_dict={x: X_t_batch,
                                                      y: Y_t_batch})
                v_precision = session.run(accuracy, feed_dict={x: X_v_batch,
                                                               y: Y_v_batch})
                v_cost = session.run(loss, feed_dict={x: X_v_batch,
                                                      y: Y_v_batch})
                train = session.run(train_op, feed_dict={x: X_t_batch,
                                                         y: Y_t_batch})
                j += batch_size
                if k % 100 == 0 and k != 0:
                    print("\tStep {}:".format(k))
                    print("\t\tCost: {}".format(t_cost))
                    print("\t\tAccuracy: {}".format(t_precision))
            print("After {} epochs".format(i))
            print("\tTraining Cost: {}".format(t_cost))
            print("\tTraining Accuracy: {}".format(t_precision))
            print("\tTraining Cost: {}".format(v_cost))
            print("\tTraining Accuracy: {}".format(v_precision))
        return saver.save(session, save_path)