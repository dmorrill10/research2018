import tensorflow as tf


def unnormalized_negative_entropy(y, y_hat):
    '''Matching loss for an exponential activation function.'''
    cross_entropy_terms = tf.where(
        tf.greater(y, 0.0), y * tf.log(y_hat), tf.zeros_like(y))
    return tf.reduce_sum(y_hat - cross_entropy_terms, axis=1)


def mean_unnormalized_negative_entropy(*args, **kwargs):
    return tf.reduce_mean(unnormalized_negative_entropy(*args))


def cross_entropy(y, y_hat):
    '''Matching loss for a sigmoid activation function.'''
    y = tf.convert_to_tensor(y)
    y_hat = tf.convert_to_tensor(y_hat)
    positive_cross_entropy_terms = tf.where(
        tf.greater(y, 0.0), y * tf.log(y_hat), tf.zeros_like(y))
    negative_cross_entropy_terms = tf.where(
        tf.less(y, 1.0), (1.0 - y) * tf.log(1.0 - y_hat), tf.zeros_like(y))
    return -tf.reduce_sum(
        positive_cross_entropy_terms + negative_cross_entropy_terms, axis=1)


def mean_cross_entropy(*args, **kwargs):
    return tf.reduce_mean(cross_entropy(*args))
