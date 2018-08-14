import numpy as np


class PlateauFunction(object):
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

    def with_function_outside_plateaus(self, f):
        return self.__class__(self.heights, self.x_clusters, f)
