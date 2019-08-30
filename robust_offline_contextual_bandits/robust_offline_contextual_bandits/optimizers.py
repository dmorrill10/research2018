import tensorflow as tf
import numpy as np


def new_t_inv_gd_optimizer(learning_rate):
    return tf.train.GradientDescentOptimizer(
        learning_rate=tf.train.inverse_time_decay(
            learning_rate, tf.train.get_global_step(), 1, 1))


def sum_over_dims(t):
    return tf.reduce_sum(t, axis=0, keep_dims=True)


def tile_to_dims(t, num_dims):
    return tf.tile(t, [num_dims, 1])


def with_fixed_dimensions(t,
                          independent_dimensions=False,
                          dependent_columns=False):
    if dependent_columns:
        return tf.reshape(t, [tf.size(t), 1])
    elif len(t.shape) < 2:
        return tf.expand_dims(t, axis=0)
    else:
        num_columns = t.shape[-1].value
        num_dimensions = (
            np.prod([t.shape[i].value for i in range(len(t.shape) - 1)])
            if len(t.shape) > 2 else
            t.shape[0].value
        )  # yapf:disable
        if independent_dimensions:
            num_columns = num_dimensions * num_columns
            num_dimensions = 1
        return tf.reshape(t, [num_dimensions, num_columns])
