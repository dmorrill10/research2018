import tensorflow as tf
import numpy as np
from tensorflow.python.ops.resource_variable_ops import ResourceVariable
from tf_contextual_prediction_with_expert_advice import \
    rm_policy, \
    utility, \
    normalized


def linear_avg_next_policy_sum(policy_sum, _cur, t):
    return policy_sum + t * _cur


def uniform_avg_next_policy_sum(policy_sum, _cur, t):
    return policy_sum + _cur


def exp_avg_next_policy_sum(policy_sum, _cur, t, alpha=0.9):
    return alpha * policy_sum + (1.0 - alpha) * _cur


class TabularCfr(object):
    @classmethod
    def zeros(cls, num_info_sets, num_actions, *args, **kwargs):
        return cls(
            tf.zeros([num_info_sets, num_actions]),
            tf.zeros([num_info_sets, num_actions]), *args, **kwargs)

    @classmethod
    def load(cls, name):
        return cls(*np.load('{}.npy'.format(name)))

    def __init__(self, regrets, policy_sum, t=0):
        self.regrets = ResourceVariable(regrets)
        self.policy_sum = ResourceVariable(policy_sum)
        self.t = ResourceVariable(t)

    def save(self, name):
        np.save(name, [self.regrets, self.policy_sum, self.t])
        return self

    def graph_save(self, name, sess):
        np.save(name, sess.run([self.regrets, self.policy_sum, self.t]))
        return self

    def num_info_sets(self):
        return tf.shape(self.regrets)[0]

    def num_actions(self):
        return tf.shape(self.regrets)[1]

    def reset(self):
        for v in [self.regrets, self.policy_sum]:
            v.assign(tf.zeros_like(v))
        self.t.assign(0)

    def copy(self):
        return self.__class__(self.regrets, self.policy_sum)

    def cur(self):
        return rm_policy(self.regrets)

    def avg(self):
        return normalized(self.policy_sum)

    def policy(self, mix_avg=1.0):
        use_cur = mix_avg < 1
        pol = 0.0
        if use_cur:
            cur = self.cur()
            pol = (1.0 - mix_avg) * cur

        use_avg = mix_avg > 0
        if use_avg:
            avg = self.avg()
            pol = mix_avg * avg + pol
        return pol

    def update(self, env, mix_avg=0.0, rm_plus=False, next_policy_sum=None):
        cur = self.cur()
        policy = (1.0 - mix_avg) * cur

        use_avg = mix_avg > 0
        if use_avg:
            avg = self.avg()
            policy = policy + mix_avg * avg

        cfv = env(policy)

        evs = utility(cur, cfv)
        regrets = cfv - evs

        update_t = self.t.assign_add(1)
        if rm_plus:
            if next_policy_sum is None:
                next_policy_sum = linear_avg_next_policy_sum
            update_regrets = self.regrets.assign(
                tf.nn.relu(self.regrets + regrets))
        else:
            if next_policy_sum is None:
                next_policy_sum = uniform_avg_next_policy_sum
            update_regrets = self.regrets.assign_add(regrets)

        update_policy_sum = self.policy_sum.assign(
            next_policy_sum(self.policy_sum, cur,
                            tf.cast(self.t + 1, tf.float32)))
        return evs, tf.group(update_policy_sum, update_regrets, update_t)


class FixedParameterCfr(object):
    @classmethod
    def load(cls, name, cfr_cls=TabularCfr):
        return cls(*np.load('{}.params.npy'.format(name)),
                   cfr_cls.load('{}.cfr'.format(name)))

    def __init__(self, use_plus, mix_avg, cfr):
        self.cfr = cfr
        self.use_plus = use_plus
        self.mix_avg = mix_avg

    def params(self):
        return [self.use_plus, self.mix_avg]

    def save(self, name):
        np.save('{}.params'.format(name), self.params())
        self.cfr.save('{}.cfr'.format(name))
        return self

    def graph_save(self, name, sess):
        np.save('{}.params'.format(name), self.params())
        self.cfr.graph_save('{}.cfr'.format(name), sess)
        return self

    def next_policy_sum(self):
        raise NotImplementedError('Please override')

    def update(self, env):
        return self.cfr.update(
            env,
            rm_plus=self.use_plus,
            mix_avg=self.mix_avg,
            next_policy_sum=self.next_policy_sum())


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
