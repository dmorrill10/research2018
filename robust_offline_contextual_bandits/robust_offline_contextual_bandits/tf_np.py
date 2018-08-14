import tensorflow as tf
import numpy as np


def enable_eager(self):
    try:
        tf.enable_eager_execution()
    except:
        pass
    assert tf.executing_eagerly()


def reset_random(seed=42):
    tf.set_random_seed(seed)
    np.random.seed(seed)


def huber(y, knot=1.0):
    abs_y = tf.abs(y)
    return tf.where(
        tf.greater(abs_y, knot), 2 * knot * (abs_y - knot), tf.square(y))
