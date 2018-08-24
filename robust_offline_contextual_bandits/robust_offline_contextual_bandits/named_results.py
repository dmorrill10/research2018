import numpy as np


class NamedResults(object):
    def __init__(self, evs, style={}, area=None):
        self.evs = evs
        self.style = style
        self.area = area

    def avg_evs(self):
        return self.evs.mean(axis=-1) if self.num_reps() > 1 else self.evs

    def percentile_normalization(self):
        return self.num_evs() / 100.0

    def percentile_points(self):
        return np.arange(self.num_evs()) / self.percentile_normalization()

    def num_evs(self):
        return len(self.evs)

    def num_reps(self):
        return self.evs.shape[1] if self.evs.ndim > 1 else 1

    def show_area(self):
        return self.area is not None

    def min_ev(self):
        return min(self.avg_evs())

    def area_beneath(self, min_ev=None):
        if min_ev is None:
            min_ev = self.min_ev()
        lb, ub = self.area
        lb_frac = int(np.ceil(self.num_evs() * lb))
        ub_frac = int(np.ceil(self.num_evs() * ub))

        width = np.arange(lb_frac, ub_frac) / self.percentile_normalization()
        lb = np.full([ub_frac - lb_frac], min_ev)
        ub = self.avg_evs()[lb_frac:ub_frac]
        return width, lb, ub
