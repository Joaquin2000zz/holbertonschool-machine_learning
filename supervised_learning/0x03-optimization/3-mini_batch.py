#!/usr/bin/env python3
"""
module which contains shuffle_data function
"""
import tensorflow.compat.v1 as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid,
                     batch_size=32, epochs=5,
                     load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """
    trains a loaded neural network model
    using mini-batch gradient descent:
    * X_train contains the training data
    * Y_train contains the training labels
    * X_valid contains the validation data
    * Y_valid is a one-hot containing the validation labels
    * batch_size is the number of data points in a batch
    * epochs: number of times the training pass through the whole dataset
    * load_path is the path from which to load the model
    * save_path = where the model should be saved after training
    * Returns: the path where the model was saved
    """
    with tf.Session() as sess:
        # this is used to open the metadata of a training
        saver = tf.train.import_meta_graph(load_path + ".meta")
        saver.restore(sess, load_path)

        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        accuracy = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]
        train_op = tf.get_collection("train_op")[0]

        m = X_train.shape[0]
        # avoid border case if whole batch size is odd
        # // perform floor division
        if m % batch_size == 0:
            n_batches = m // batch_size
        else:
            n_batches = m // batch_size + 1

        for i in range(epochs + 1):
            t_cost = sess.run(loss, feed_dict={x: X_train, y: Y_train})
            t_precision = sess.run(accuracy,
                                   feed_dict={x: X_train, y: Y_train})
            v_cost = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
            v_precision = sess.run(accuracy,
                                   feed_dict={x: X_valid, y: Y_valid})
            print("After {} epochs:".format(i))
            print("\tTraining Cost: {}".format(t_cost))
            print("\tTraining Accuracy: {}".format(t_precision))
            print("\tValidation Cost: {}".format(v_cost))
            print("\tValidation Accuracy: {}".format(v_precision))

            if i < epochs:
                shuffled_X, shuffled_Y = shuffle_data(X_train, Y_train)

                # mini batches
                for j in range(n_batches):
                    start = j * batch_size
                    end = (j + 1) * batch_size
                    if end > m:
                        end = m
                    X_mini_batch = shuffled_X[start:end]
                    Y_mini_batch = shuffled_Y[start:end]

                    sess.run(train_op, feed_dict={x: X_mini_batch,
                                                  y: Y_mini_batch})

                    if (j + 1) % 100 == 0 and j != 0:
                        t_cost = sess.run(loss, feed_dict={x: X_mini_batch,
                                                           y: Y_mini_batch})
                        t_precision = sess.run(accuracy,
                                               feed_dict={x: X_mini_batch,
                                                          y: Y_mini_batch})
                        print("\tStep {}:".format(j + 1))
                        print("\t\tCost: {}".format(t_cost))
                        print("\t\tAccuracy: {}".format(t_precision))

        return saver.save(sess, save_path)
