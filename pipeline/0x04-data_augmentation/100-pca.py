def pca_color(image, alphas=None):
    """
    performs PCA color augmentation as described in the AlexNet paper:

    - image: is a 3D tf.Tensor containing the image to change
    - alphas: a tuple of length 3 containing the amount that
    each channel should change
    Returns: the augmented image
    """
    # convert the image tensor to a 2D tensor
    h, w, c = image.shape
    flattened_image = tf.reshape(image, [-1, c])
    # subtract the mean from each pixel
    mean = tf.reduce_mean(flattened_image, axis=0)
    centered_image = tf.cast(image - mean, tf.float32) 
    # compute covariance matrix
    covariance = tf.matmul(centered_image, centered_image,
                           transpose_a=True) / tf.cast(h * w, tf.float32)
    # compute the eigenvectors and eigenvalues of the covariance matrix
    eigenvalues, eigenvectors = tf.linalg.eigh(covariance)
    # sort the eigenvectors by descending eigenvalues
    sorted_indices = tf.argsort(eigenvalues,
                                direction='DESCENDING')
    sorted_eigenvectors = tf.gather(eigenvectors, sorted_indices, axis=1)
    # compute the projection matrix and project the centered image onto it
    projection_matrix = sorted_eigenvectors[:, :c]
    if None not in alphas:
      projection_matrix = tf.multiply(projection_matrix, alphas) 
    pca_image = tf.matmul(centered_image, projection_matrix)
    pca_image = tf.reshape(pca_image, [h, w, c])
    return pca_image
