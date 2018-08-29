import tensorflow as tf


def new_t_inv_gd_optimizer(learning_rate):
    return tf.train.GradientDescentOptimizer(
        learning_rate=tf.train.inverse_time_decay(
            learning_rate, tf.train.get_global_step(), 1, 1))
