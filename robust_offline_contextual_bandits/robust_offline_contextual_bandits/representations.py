import tensorflow as tf
import numpy as np
from robust_offline_contextual_bandits.tile_coding import \
    tile_coding_dense_feature_expansion


class RepresentationWithFixedInputs(object):
    def __init__(self, phi_f, x):
        self.phi_f = phi_f
        self.x = x
        self.phi = phi_f(x)

    def num_examples(self):
        return self.phi.shape[0].value

    def num_features(self):
        return self.phi.shape[1].value

    def input_generator(self):
        yield self.phi


class RawRepresentationWithFixedInputs(RepresentationWithFixedInputs):
    def __init__(self, x):
        super(RawRepresentationWithFixedInputs, self).__init__(
            lambda x: tf.convert_to_tensor(x), x)


class TileCodingRepresentationWithFixedInputs(RepresentationWithFixedInputs):
    def __init__(self, num_tiling_pairs, x, tile_width_fractions=None):
        self.num_tiling_pairs = num_tiling_pairs
        bounds = list(zip(np.min(x, axis=0), np.max(x, axis=0)))
        _phi_f, _ = tile_coding_dense_feature_expansion(
            bounds, num_tiling_pairs, tile_width_fractions)

        def phi_f(x):
            return (tf.stack([_phi_f(state).astype('float32') for state in x])
                    if len(x) > 0 else tf.convert_to_tensor(
                        _phi_f(x).astype('float32')))

        super(TileCodingRepresentationWithFixedInputs, self).__init__(phi_f, x)

    def learning_rate(self):
        return float(self.num_examples()) / (2 * self.num_tiling_pairs + 1.0)


class TabularRepresentationWithFixedInputs(RepresentationWithFixedInputs):
    def __init__(self, x):
        super(TabularRepresentationWithFixedInputs, self).__init__(
            lambda x: tf.eye(len(x)), x)

    def learning_rate(self):
        return float(self.num_examples())


class LiftAndProjectRepresentationWithFixedInputs(
        RepresentationWithFixedInputs):
    def __init__(self, x):
        def phi_f(x):
            x = tf.convert_to_tensor(x)
            if len(x.shape) < 2:
                x = tf.expand_dims(x, axis=1)
            norm = tf.norm(x, axis=1, keepdims=True)
            return tf.concat([x, tf.ones([tf.shape(x)[0], 1])], axis=1) / norm

        super(LiftAndProjectRepresentationWithFixedInputs, self).__init__(
            phi_f, x)
