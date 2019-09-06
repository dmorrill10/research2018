import tensorflow as tf
import numpy as np
import os
from glob import glob
from tf_supervised_inference.data import Data, NamedDataSets
from research2018 import rrm

from robust_offline_contextual_bandits.data import DataComponentsForTraining


def training_data(plateau_functions, x, stddev=0.0):
    for f in plateau_functions:
        yield f.training_data(x, stddev=stddev)


def slope_and_bias_across_constants_for_unknown_outputs(
        plateau_functions, x, policy):
    test_rewards = np.array([[
        f(x, outside_plateaus=lambda x: np.full([len(x)], float(i)))
        for f in plateau_functions
    ] for i in range(2)])
    bias = tf.reduce_mean(rrm.utility(policy, test_rewards[0].T))
    slope = tf.reduce_mean(rrm.utility(policy, test_rewards[1].T)) - bias
    return slope, bias


def _bounds(x):
    min_x, max_x = min(x), max(x)
    diff = max_x - min_x
    remaining_diff = 2 / 500.0 - diff
    if remaining_diff > 0:
        min_x -= remaining_diff / 2.0
        max_x += remaining_diff / 2.0
    return min_x, max_x


class PlateauFunction(object):
    @classmethod
    def load(cls, name):
        return cls(*np.load('{}.npy'.format(name)))

    @classmethod
    def load_all(cls, pattern):
        return {
            file: cls.load(os.path.splitext(file)[0])
            for file in glob(pattern)
        }

    def __init__(self, heights, bounds):
        self.heights = heights
        self.bounds = bounds

    @property
    def centers(self):
        return [sum(b) / 2.0 for b in self.bounds]

    def save(self, name):
        np.save('{}.npy'.format(name),
                np.array([self.heights, self.bounds], dtype=object))
        return self

    def __call__(self, x, outside_plateaus=None, stddev=0.0):
        if outside_plateaus is None:
            y = np.full([len(x)], np.nan)
        else:
            y = outside_plateaus(x)
        if len(x.shape) < 2:
            x = np.expand_dims(x, 1)
        for i, (x_min, x_max) in enumerate(self.bounds):
            x_in_bounds = np.logical_and(np.all(x_min <= x, axis=-1),
                                         np.all(x <= x_max, axis=-1))
            num_bounded_x = x_in_bounds.sum()
            y[x_in_bounds] = np.random.normal(self.heights[i],
                                              stddev,
                                              size=[num_bounded_x])
        if outside_plateaus is None:
            return y[np.isfinite(y)]
        else:
            return y

    def training_data(self, x, stddev=0.0):
        x_train = x[self.in_bounds(x)]
        y_train = self(x_train, stddev=stddev)
        assert len(x_train) == len(y_train)
        return x_train, y_train

    def in_bounds(self, x):
        if len(x.shape) < 2:
            x = np.expand_dims(x, 1)
        in_bounds = np.full([len(x)], False)
        for i, (x_min, x_max) in enumerate(self.bounds):
            np.logical_or(in_bounds,
                          np.logical_and(np.all(x_min <= x, axis=-1),
                                         np.all(x <= x_max, axis=-1)),
                          out=in_bounds)
        return in_bounds

    def for_training(self,
                     x,
                     stddev=0.0,
                     outside_plateaus=lambda x: np.zeros([len(x)])):
        y = self(np.squeeze(x),
                 outside_plateaus=outside_plateaus).astype('float32')

        ge = self.in_bounds(x)
        be = np.logical_not(ge)

        gdata = Data(x[ge], np.expand_dims(y[ge], axis=1))
        bdata = Data(x[be], np.expand_dims(y[be], axis=1))
        data = NamedDataSets(good=gdata, bad=bdata)

        noisy_y = np.expand_dims(self(np.squeeze(x), stddev=stddev),
                                 axis=1).astype('float32')

        gdata = Data(x[ge], noisy_y[ge])
        bdata = Data(x[be], noisy_y[be])
        noisy_data = NamedDataSets(good=gdata, bad=bdata)

        combined_raw_data = data.all()
        sort_indices = combined_raw_data.phi.numpy().argsort(axis=0).squeeze()

        return DataComponentsForTraining(data, noisy_data, combined_raw_data,
                                         sort_indices)


class PlateauFunctionDistribution(object):
    def sample(self):
        num_plateaus = int(np.ceil(self.sample_num_plateaus()))
        heights = [self.sample_height() for _ in range(num_plateaus)]
        centers = [self.sample_center() for _ in range(num_plateaus)]
        radii = [self.sample_radius() for _ in range(num_plateaus)]
        bounds = [(c - radii[i], c + radii[i]) for i, c in enumerate(centers)]
        return PlateauFunction(heights, bounds)

    def sample_num_plateaus(self):
        raise NotImplementedError('Override')

    def sample_height(self):
        raise NotImplementedError('Override')

    def sample_center(self):
        raise NotImplementedError('Override')

    def sample_radius(self):
        raise NotImplementedError('Override')
