import tensorflow as tf
import numpy as np


def reset_random(seed=42):
    tf.random.set_seed(seed)
    np.random.seed(seed)


def huber(y, knot=1.0):
    abs_y = tf.abs(y)
    return tf.where(tf.greater(abs_y, knot), 2 * knot * (abs_y - knot),
                    tf.square(y))


def save_model_to_yml(model, name):
    model_yaml = model.to_yaml()
    with open('{}.yml'.format(name), "w") as yaml_file:
        yaml_file.write(model_yaml)
    model.save_weights('{}.h5'.format(name))


def load_model_from_yml(name):
    with open('{}.yml'.format(name), 'r') as yaml_file:
        loaded_model_yaml = yaml_file.read()
    loaded_model = tf.keras.models.model_from_yaml(loaded_model_yaml)
    loaded_model.load_weights('{}.h5'.format(name))
    return loaded_model


def logical_or(*a):
    v = np.full(a[0].shape, False)
    for ai in a:
        np.logical_or(v, ai, v)
    return v
