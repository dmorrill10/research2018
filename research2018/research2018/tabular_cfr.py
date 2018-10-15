import tensorflow as tf
import numpy as np
from tensorflow.python.ops.resource_variable_ops import ResourceVariable
from tf_contextual_prediction_with_expert_advice import rm_policy, utility


class TabularCfr(object):
    @classmethod
    def zeros(cls, num_info_sets, num_actions, **kwargs):
        return cls(
            tf.zeros([num_info_sets, num_actions]),
            tf.zeros([num_info_sets, num_actions]), **kwargs)

    @classmethod
    def load(cls, name):
        return cls(*np.load('{}.npy'.format(name)))

    def __init__(self, regrets, policy_sum, t=0):
        self.regrets = ResourceVariable(regrets)
        self.policy_sum = ResourceVariable(policy_sum)
        self.t = t
        self._last_t_cur = t - 1
        self._last_t_avg = t - 1
        self._cur = None
        self._avg = None

    def save(self, name):
        np.save(name, [self.regrets, self.policy_sum, self.t])
        return self

    def num_info_sets(self):
        return tf.shape(self.regrets)[0]

    def num_actions(self):
        return tf.shape(self.regrets)[1]

    def reset(self):
        for v in [self.regrets, self.policy_sum]:
            v.assign(tf.zeros_like(v))
        self.t = 0

    def copy(self):
        return self.__class__(self.regrets, self.policy_sum)

    def cur(self):
        if self.t > self._last_t_cur:
            self._last_t_cur = self.t
            self._cur = rm_policy(self.regrets)
        return self._cur

    def avg(self):
        if self.t > self._last_t_avg:
            self._last_t_avg = self.t
            self._avg = rm_policy(self.policy_sum)
        return self._avg

    def policy(self, mix_avg=0.0):
        use_cur = mix_avg < 1
        pol = 0.0
        if use_cur:
            cur = self.cur()
            pol = (1.0 - mix_avg) * cur

        use_avg = mix_avg > 0
        if use_avg:
            avg = self.avg()
            pol = pol + mix_avg * avg
        return pol

    def __call__(self, env, mix_avg=0.0):
        return env(self.policy(mix_avg))

    def ev(self, env, context_weights=None, mix_avg=0.0):
        policy = self.policy(mix_avg)
        cfv = env(policy)

        context_values = utility(policy, cfv)
        if context_weights is not None:
            context_weights = tf.convert_to_tensor(context_weights)
            if len(context_weights.shape) < len(context_values.shape):
                extra_dims = (
                    len(context_values.shape) - len(context_weights.shape))
                context_weights = tf.reshape(
                    context_weights,
                    ([s.value
                      for s in context_weights.shape] + [1] * extra_dims))
            context_values = context_weights * context_values
            return tf.reduce_sum(context_values)
        else:
            return tf.reduce_mean(context_values)

    def update(self, env, mix_avg=0.0, rm_plus=False, for_avg=None):
        cur = self.cur()
        policy = (1.0 - mix_avg) * cur

        use_avg = mix_avg > 0
        if use_avg:
            avg = self.avg()
            policy = policy + mix_avg * avg

        cfv = env(policy)

        evs = utility(cur, cfv)
        regrets = cfv - evs

        self.t += 1
        if rm_plus:
            if for_avg is None:

                def for_avg(_cur, t):
                    return t * _cur  # Linear avg

            update_regrets = self.regrets.assign(
                tf.nn.relu(self.regrets + regrets))
        else:
            if for_avg is None:

                def for_avg(_cur, t):
                    return _cur  # Uniform avg

            update_regrets = self.regrets.assign_add(regrets)

        update_policy_sum = self.policy_sum.assign_add(for_avg(cur, self.t))
        return evs, update_policy_sum, update_regrets
