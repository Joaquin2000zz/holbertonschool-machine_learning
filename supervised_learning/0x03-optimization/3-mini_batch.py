#!/usr/bin/env python3
"""
module which contains shuffle_data function
"""
shuffle_data = __import__('2-shuffle_data').shuffle_data
import tensorflow.compat.v1 as tf


def train_mini_batch(X_train, Y_train, X_valid, Y_valid,
                     batch_size=32, epochs=5,
                     load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """
    trains a loaded neural network model using mini-batch gradient descent:
    * X_train contains the training data
    * Y_train contains the training labels
    * X_valid contains the validation data
    * Y_valid is a one-hot containing the validation labels
    * batch_size is the number of data points in a batch
    * epochs: number of times the training pass through the whole dataset
    * load_path is the path from which to load the model
    * save_path is the path to where the model should
      be saved after training
    * Returns: the path where the model was saved
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
    """
    with tf.Session() as session:
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

        # instance of tf.train.Saver() to save
        saver = tf.train.Saver()

        l_batch = int(X_train.shape[0] / batch_size)
        for i in range(epochs):
            t_cost = session.run(loss, feed_dict={x: X_train, y: Y_train})
            t_precision = session.run(accuracy, feed_dict={x: X_train,
                                                           y: Y_train})

            v_cost = session.run(loss, feed_dict={x: X_valid, y: Y_valid})
            v_precision = session.run(accuracy, feed_dict={x: X_valid,
                                                           y: Y_valid})

            print("After {} epochs".format(i))
            print("\tTraining Cost: {}".format(t_cost))
            print("\tTraining Accuracy: {}".format(t_precision))
            print("\tTraining Cost: {}".format(v_cost))
            print("\tTraining Accuracy: {}".format(v_precision))

            X_shuffle, Y_shuffle = shuffle_data(X_train, Y_train)

            for j in range(l_batch):
                start = j * batch_size
                end = (j + 1) * batch_size

                if end > X_train.shape[0]:
                    end = X_train.shape[0]

                X_t_batch = X_shuffle[start: end]
                Y_t_batch = Y_shuffle[start: end]
                X_t_batch, Y_t_batch = shuffle_data(X_t_batch, Y_t_batch)

                session.run(train_op, feed_dict={x: X_t_batch,
                                                 y: Y_t_batch})

                if j + 1 % 100 == 0 and j != 0:
                    t_precision = session.run(accuracy,
                                              feed_dict={x: X_t_batch,
                                                         y: Y_t_batch})
                    t_cost = session.run(loss, feed_dict={x: X_valid,
                                                          y: Y_valid})

                    print("\tStep {}:".format(j + 1))
                    print("\t\tCost: {}".format(t_cost))
                    print("\t\tAccuracy: {}".format(t_precision))
        return saver.save(session, save_path)
