import numpy as np
from research2018.tabular_cfr import \
    linear_avg_next_policy_sum, \
    uniform_avg_next_policy_sum, \
    exp_avg_next_policy_sum, \
    TabularCfr


class FixedParameterCfr(object):
    @classmethod
    def load(cls, name, cfr_cls=TabularCfr):
        return cls(
            cfr=cfr_cls.load('{}.cfr'.format(name)),
            **np.load('{}.params.npy'.format(name)))

    def __init__(self, cfr, use_plus=False, mix_avg=None):
        self.cfr = cfr
        self.use_plus = use_plus
        self.mix_avg = mix_avg

    def params(self):
        return {'use_plus': self.use_plus, 'mix_avg': self.mix_avg}

    def save(self, name):
        np.save('{}.params'.format(name), self.params())
        self.cfr.save('{}.cfr'.format(name))
        return self

    def graph_save(self, name, sess):
        np.save('{}.params'.format(name), self.params())
        self.cfr.graph_save('{}.cfr'.format(name), sess)
        return self

    def next_policy_sum(self):
        return None

    def update(self, env):
        kwargs = {'rm_plus': self.use_plus}
        if self.mix_avg is not None:
            kwargs['mix_avg'] = self.mix_avg
        if self.next_policy_sum() is not None:
            kwargs['next_policy_sum'] = self.next_policy_sum()
        return self.cfr.update(env, **kwargs)


class FixedParameterAvgCodeCfr(FixedParameterCfr):
    USE_UNIFORM_AVG = -2
    USE_LINEAR_AVG = -1

    def __init__(self, next_policy_sum_code, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.next_policy_sum_code = next_policy_sum_code

    def params(self):
        return [self.next_policy_sum_code] + super().params()

    def next_policy_sum(self):
        if self.next_policy_sum_code == self.USE_UNIFORM_AVG:
            return uniform_avg_next_policy_sum
        elif self.next_policy_sum_code == self.USE_LINEAR_AVG:
            return linear_avg_next_policy_sum
        else:

            def f(*args):
                return exp_avg_next_policy_sum(
                    *args, alpha=self.next_policy_sum_code)

            return f
