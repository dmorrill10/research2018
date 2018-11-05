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


class TabularCfrCurrent(object):
    @classmethod
    def zeros(cls, num_info_sets, num_actions, *args, **kwargs):
        return cls(tf.zeros([num_info_sets, num_actions]), *args, **kwargs)

    @classmethod
    def load(cls, name):
        return cls(*np.load('{}.npy'.format(name)))

    def __init__(self, regrets):
        self.regrets = ResourceVariable(regrets)
        self._has_updated = False
        self._policy = rm_policy(self.regrets)

    def save(self, name):
        np.save(name, self.regrets)
        return self

    def graph_save(self, name, sess):
        np.save(name, sess.run(self.regrets))
        return self

    def num_info_sets(self):
        return tf.shape(self.regrets)[0]

    def num_actions(self):
        return tf.shape(self.regrets)[1]

    def clear(self):
        self.regrets.assign(tf.zeros_like(self.regrets))

    def copy(self):
        return self.__class__(self.regrets)

    def policy(self):
        if tf.executing_eagerly() and self._has_updated:
            self._policy = rm_policy(self.regrets)
            self._has_updated = False
        return self._policy

    def update(self, env, **kwargs):
        return self.update_with_cfv(env(self.policy()), **kwargs)

    def update_with_cfv(self, cfv, rm_plus=False):
        evs = utility(self.policy(), cfv)
        regrets = cfv - evs

        r = self.regrets + regrets
        if rm_plus:
            r = tf.nn.relu(r)
        self._has_updated = True
        return evs, self.regrets.assign(r)


class TabularCfr(object):
    @classmethod
    def zeros(cls, num_info_sets, num_actions, *args, **kwargs):
        return cls(
            TabularCfrCurrent.zeros(num_info_sets, num_actions),
            tf.zeros([num_info_sets, num_actions]), *args, **kwargs)

    @classmethod
    def load(cls, name):
        return cls(
            TabularCfrCurrent.load('{}.cur'.format(name)),
            *np.load('{}.npy'.format(name)))

    def __init__(self, cur, policy_sum, t=0):
        self._cur = cur
        self.policy_sum = ResourceVariable(policy_sum)
        self.t = ResourceVariable(t)

    def save(self, name):
        self._cur.save('{}.cur'.format(name))
        np.save(name, [self.policy_sum, self.t])
        return self

    def graph_save(self, name, sess):
        self._cur.graph_save('{}.cur'.format(name), sess)
        np.save(name, sess.run([self.policy_sum, self.t]))
        return self

    @property
    def num_info_sets(self):
        return self._cur.num_info_sets

    @property
    def num_actions(self):
        return self._cur.num_actions

    def clear(self):
        self._cur.clear()
        self.policy_sum.assign(tf.zeros_like(self.policy_sum))
        self.t.assign(0)

    def copy(self, copy_t=False):
        if copy_t:
            return self.__class__(self._cur.copy(), self.policy_sum, self.t)
        else:
            return self.__class__(self._cur.copy(), self.policy_sum)

    @property
    def cur(self):
        return self._cur.policy

    def avg(self):
        return normalized(self.policy_sum, axis=1)

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

        evs, update_current = self._cur.update_with_cfv(
            env(policy), rm_plus=rm_plus)
        update_t = self.t.assign_add(1)

        if next_policy_sum is None:
            next_policy_sum = linear_avg_next_policy_sum if rm_plus else uniform_avg_next_policy_sum

        update_policy_sum = self.policy_sum.assign(
            next_policy_sum(self.policy_sum, cur,
                            tf.cast(self.t + 1, tf.float32)))
        return evs, tf.group(update_policy_sum, update_current, update_t)
