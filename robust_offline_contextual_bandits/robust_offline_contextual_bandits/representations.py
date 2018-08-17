import tensorflow as tf
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
        super(RawRepresentationWithFixedInputs,
              self).__init__(lambda x: tf.convert_to_tensor(x))


class TileCodingRepresentationWithFixedInputs(RepresentationWithFixedInputs):
    def __init__(self, bounds, num_tilings, num_tiles, x):
        self.num_tilings = num_tilings
        self.num_tiles = num_tiles
        _phi_f, _ = tile_coding_dense_feature_expansion(
            bounds, num_tilings, num_tiles)

        def phi_f(x):
            return (tf.stack([_phi_f(state).astype('float32') for state in x])
                    if len(x) > 0 else tf.convert_to_tensor(
                        _phi_f(x).astype('float32')))

        super(TileCodingRepresentationWithFixedInputs, self).__init__(phi_f, x)

    def learning_rate(self):
        return float(self.num_tiles) / self.num_tilings


class TabularRepresentationWithFixedInputs(
        TileCodingRepresentationWithFixedInputs):
    def __init__(self, bounds, x):
        super(TabularRepresentationWithFixedInputs, self).__init__(
            bounds, 1, len(x), x)
