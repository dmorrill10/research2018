import numpy as np
from robust_offline_contextual_bandits.data import DataComponentsForTraining
from tf_supervised_inference.data import Data, NamedDataSets


class PlateauFunction(object):
    @classmethod
    def sample_from_bounds_and_averages(cls, x_min, x_max, avg_num_plateaus,
                                        avg_num_points_per_plateau,
                                        function_outside_plateaus):
        stddev = np.abs(x_max - x_min)
        cluster_stddev = stddev / 20.0
        num_plateaus = int(
            np.ceil(np.abs(np.random.normal(avg_num_plateaus, stddev))))
        heights = np.random.normal(0.0, size=[num_plateaus])
        midpoints = np.random.uniform(x_min, x_max, size=[num_plateaus])
        num_points_per_plateau = max(
            1,
            int(
                np.ceil(
                    np.abs(
                        np.random.normal(avg_num_points_per_plateau,
                                         stddev)))))
        return cls.sample(midpoints, heights, num_points_per_plateau,
                          cluster_stddev, function_outside_plateaus)

    @classmethod
    def sample(cls, centers, heights, num_points_per_plateau, cluster_stddev,
               function_outside_plateaus):
        x_clusters = [
            m +
            np.random.normal(0, cluster_stddev, size=[num_points_per_plateau])
            for m in centers
        ]
        return cls(heights, x_clusters, function_outside_plateaus)

    def __init__(self, heights, x_clusters, function_outside_plateaus):
        self.heights = heights
        self.function_outside_plateaus = function_outside_plateaus
        self.x_clusters = x_clusters
        self.x_bounds = [(min(x), max(x)) for x in self.x_clusters]

    def __call__(self, x, stddev=0.0):
        y = np.random.normal(
            self.function_outside_plateaus(x), stddev, size=[len(x)])
        for i in range(len(self.x_bounds)):
            x_min, x_max = self.x_bounds[i]
            x_in_bounds = np.logical_and(x_min <= x, x <= x_max)
            num_bounded_x = x_in_bounds.sum()
            y[x_in_bounds] = np.random.normal(
                self.heights[i], stddev, size=[num_bounded_x])
        return y

    def in_bounds(self, x):
        x = x.squeeze()
        in_bounds = np.full(x.shape, False)
        for i in range(len(self.x_bounds)):
            x_min, x_max = self.x_bounds[i]
            np.logical_or(
                in_bounds,
                np.logical_and(x_min <= x, x <= x_max),
                out=in_bounds)
        return in_bounds

    def with_function_outside_plateaus(self, f):
        return self.__class__(self.heights, self.x_clusters, f)

    def for_training(self, x, stddev=0.0):
        y = self(np.squeeze(x)).astype('float32')

        ge = self.in_bounds(x)
        be = np.logical_not(ge)

        gdata = Data(x[ge], np.expand_dims(y[ge], axis=1))
        bdata = Data(x[be], np.expand_dims(y[be], axis=1))
        data = NamedDataSets(good=gdata, bad=bdata)

        noisy_y = np.expand_dims(
            self(np.squeeze(x), stddev=stddev), axis=1).astype('float32')

        gdata = Data(x[ge], noisy_y[ge])
        bdata = Data(x[be], noisy_y[be])
        noisy_data = NamedDataSets(good=gdata, bad=bdata)

        combined_raw_data = data.all()
        sort_indices = combined_raw_data.phi.numpy().argsort(axis=0).squeeze()

        return DataComponentsForTraining(data, noisy_data, combined_raw_data,
                                         sort_indices)
