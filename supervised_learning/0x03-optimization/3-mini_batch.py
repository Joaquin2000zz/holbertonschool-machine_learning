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
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(load_path + ".meta")
        saver.restore(sess, load_path)

        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        accuracy = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]
        train_op = tf.get_collection("train_op")[0]

        m = X_train.shape[0]
        # avoid border case if batch has odd shape
        # // performs flat division
        if m % batch_size == 0:
            n_batches = m // batch_size
        else:
            n_batches = m // batch_size + 1

        for i in range(epochs + 1):
            cost_train = sess.run(loss, feed_dict={x: X_train, y: Y_train})
            accuracy_train = sess.run(accuracy,
                                      feed_dict={x: X_train, y: Y_train})
            cost_val = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
            accuracy_val = sess.run(accuracy,
                                    feed_dict={x: X_valid, y: Y_valid})
            print("After {} epochs:".format(i))
            print("\tTraining Cost: {}".format(cost_train))
            print("\tTraining Accuracy: {}".format(accuracy_train))
            print("\tValidation Cost: {}".format(cost_val))
            print("\tValidation Accuracy: {}".format(accuracy_val))

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

                    next_train = {x: X_mini_batch, y: Y_mini_batch}
                    sess.run(train_op, feed_dict=next_train)

                    if (j + 1) % 100 == 0 and j != 0:
                        loss_mini_batch = sess.run(loss, feed_dict=next_train)
                        acc_mini_batch = sess.run(accuracy,
                                                  feed_dict=next_train)
                        print("\tStep {}:".format(j + 1))
                        print("\t\tCost: {}".format(loss_mini_batch))
                        print("\t\tAccuracy: {}".format(acc_mini_batch))

        return saver.save(sess, save_path)