import tensorflow as tf
import numpy as np
from robust_offline_contextual_bandits.tile_coding import \
    tile_coding_dense_feature_expansion


class RepresentationWithFixedInputs(object):
    @classmethod
    def transform(cls, x, phi_f=lambda y: tf.convert_to_tensor(y), **kwargs):
        return cls(phi_f(x), **kwargs)

    @classmethod
    def dense_tile_coding(cls,
                          x,
                          num_tiling_pairs,
                          tile_width_fractions=None,
                          **kwargs):
        bounds = list(zip(np.min(x, axis=0), np.max(x, axis=0)))
        _phi_f, _, lr = tile_coding_dense_feature_expansion(
            bounds, num_tiling_pairs, tile_width_fractions)

        def phi_f(x):
            return (
                tf.stack([_phi_f(state).astype('float32') for state in x])
                if len(x) > 0
                else tf.convert_to_tensor(_phi_f(x).astype('float32'))
            )  # yapf:disable
        return cls(phi_f(x), lr, **kwargs)

    @classmethod
    def tabular(cls, x, **kwargs):
        return cls(tf.eye(len(x)), **kwargs)

    @classmethod
    def lift_and_project(cls, x, **kwargs):
        x = tf.convert_to_tensor(x)
        if len(x.shape) < 2:
            x = tf.expand_dims(x, axis=1)
        ones = tf.ones([tf.shape(x)[0], 1])
        x = tf.concat([x, ones], axis=1)
        return cls(x / tf.norm(x, axis=1, keepdims=True), **kwargs)

    @classmethod
    def load(cls, name):
        return cls(*np.load('{}.npy'.format(name)))

    def __init__(self, phi, learning_rate_scale=1.0):
        self.phi = phi
        self.learning_rate_scale = learning_rate_scale

    def save(self, name):
        np.save(name,
                np.array([self.phi, self.learning_rate_scale], dtype=object))
        return self

    def num_examples(self):
        return self.phi.shape[0].value

    def num_features(self):
        return self.phi.shape[1].value

    def input_generator(self):
        yield self.phi

    def learning_rate(self):
        return float(self.num_examples()) * self.learning_rate_scale
