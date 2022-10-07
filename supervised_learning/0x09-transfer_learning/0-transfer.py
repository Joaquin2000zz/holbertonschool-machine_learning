#!/usr/bin/env python3
"""
module which contains preprocess_data function
"""
import tensorflow as tf
import tensorflow.keras as K


def preprocess_data(X, Y):
    """
    pre-processes the data for the pre-trained inception_resnet_v2
    to pass to it its expected values:

    * X is a numpy.ndarray of shape (m, 32, 32, 3) containing the CIFAR 10 data,
      where m is the number of data points
    * Y is a numpy.ndarray of shape (m,) containing the CIFAR 10 labels for X
    Returns: X_p, Y_p
        - X_p is a numpy.ndarray containing the preprocessed X
        - Y_p is a numpy.ndarray containing the preprocessed Y
    """
    # preprocessing the input
    X = K.applications.inception_resnet_v2.preprocess_input(X)

    # preprocessing the expected output
    Y = K.utils.to_categorical(Y, 10)

    return X, Y


if __name__ == "__main__":

    # calling data from cifar10 images set and preprocessing it
    (X_train, Y_train), (X_valid, Y_valid) = K.datasets.cifar10.load_data()
    X_train, Y_train = preprocess_data(X=X_train, Y=Y_train)
    X_valid, Y_valid = preprocess_data(X=X_valid, Y=Y_valid)

    # transfering learning from pretrained
    # inception_resnet_v2 to our actual model
    pre_trained = K.applications.InceptionResNetV2(include_top=False, weights='imagenet',
                                                   input_shape=(299, 299, 3))

    X = K.Input(shape=(32, 32, 3))
    _, pH, pW, _ = pre_trained.input_shape
    _, xH, xW, _ = X.shape
    x = K.layers.Lambda(lambda img: tf.image.resize(img, (pH, pW)))(X)

    x = pre_trained(x, training=False)

    x = K.layers.GlobalAveragePooling2D()(x)

    x = K.layers.Dense(500, activation='relu')(x)
    x = K.layers.Dropout(0.3)(x)
    x = K.layers.Dense(10, activation='softmax')(x)

    model = K.Model(X, x)

    pre_trained.trainable = False

    model.compile(loss='categorical_crossentropy',
                  optimizer=K.optimizers.Adam(), metrics=["acc"])

    model.fit(X_train, Y_train, validation_data=(X_valid, Y_valid),
              batch_size=300, epochs=4, verbose=1)

    # now we're gonna unfreeze the second half of the pre_trained
    # model to fine-tune it we start to unfreeze starting after
    # the 402th layer (block17_8_ac) to avoid overfitting
    # also this layer is the final layer of the previous block
    for layer in pre_trained.layers[402:]:
        layer.trainable = False

    for layer in pre_trained.layers[403:]:
        layer.trainable = True

    model.compile(optimizer=K.optimizers.Adam(1e-5),
                  loss='categorical_crossentropy',
                  metrics=['acc'])

    model.fit(X_train, Y_train, validation_data=(X_valid, Y_valid),
              batch_size=300, epochs=4, verbose=1)

    model.save(r'cifar10.h5')
