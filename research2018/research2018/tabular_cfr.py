import tensorflow as tf
from tensorflow.python.ops.resource_variable_ops import ResourceVariable
from tf_contextual_bandit_rrm import rm_policy, utility


class TabularCfr(object):
    @classmethod
    def zeros(cls, num_info_sets, num_actions):
        return cls(
            ResourceVariable(tf.zeros([num_info_sets, num_actions])),
            ResourceVariable(tf.zeros([num_info_sets, num_actions])))

    def __init__(self, regrets, policy_sum):
        self.regrets = regrets
        self.policy_sum = policy_sum
        assert self.regrets.shape == self.policy_sum.shape
        self.t = 0

    def reset(self):
        for v in [self.regrets, self.policy_sum]:
            v.assign(tf.zeros_like(v))
        self.t = 0

    def copy(self):
        return self.__class__(
            ResourceVariable(self.regrets), ResourceVariable(self.policy_sum))

    def cur(self):
        return rm_policy(self.regrets)

    def avg(self):
        return rm_policy(self.policy_sum)

    def update(self, env, mix_avg=0.0, rm_plus=False):
        self.t += 1
        cur = self.cur()
        avg = self.avg()
        cfv = env(mix_avg * avg + (1 - mix_avg) * cur)
        ev = utility(cur, cfv)
        regrets = cfv - ev
        if rm_plus:
            update_policy_sum = self.policy_sum.assign_add(self.t * cur)
            update_regrets = self.regrets.assign(
                tf.nn.relu(self.regrets + regrets))
        else:
            update_policy_sum = self.policy_sum.assign_add(cur)
            update_regrets = self.regrets.assign_add(regrets)
        return tf.group(ev, update_policy_sum, update_regrets)
