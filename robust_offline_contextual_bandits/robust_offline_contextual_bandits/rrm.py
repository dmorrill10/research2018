import tensorflow as tf
from tensorflow.python.ops.resource_variable_ops import ResourceVariable
from tf_kofn_robust_policy_optimization.robust.contextual_kofn import \
    ContextualKofnGame
from research2018 import rrm
from tf_contextual_prediction_with_expert_advice.rrm import \
    rrm_loss


class RewardAugmentedMl(object):
    def policy_activation(self, pre_activations):
        return pre_activations

    def loss(self, predictions, policy, cfv):
        r = tf.nn.softmax(cfv, temp=1e-15)
        log_policy = tf.log(tf.clip_by_value(policy, 1e-15, 1 - 1e-15))
        return -tf.reduce_mean(r * log_policy)


class PolicyGradient(object):
    def policy_activation(self, pre_activations):
        return pre_activations

    def loss(self, predictions, policy, cfv):
        return -tf.reduce_mean(tf.reduce_sum(cfv * policy, axis=1))


class LinearLoss(object):
    '''Can't get this to work better than just predicting values/RRM and
    maybe it shouldn't.
    '''
    def policy_activation(self, pre_activations):
        return tf.nn.softmax(pre_activations)

    def loss(self, predictions, policy, cfv):
        return (
            -tf.reduce_mean(cfv * predictions) +
            tf.reduce_mean(tf.reduce_logsumexp(tf.abs(predictions), axis=1)))


class InstRegretStratMatching(object):
    '''Bad even though it makes some sense.'''
    def policy_activation(self, pre_activations):
        return pre_activations

    def loss(self, predictions, policy, cfv):
        r = tf.stop_gradient(
            rrm.rm_policy(cfv -
                          tf.reduce_sum(cfv * policy, axis=1, keepdims=True)))
        log_policy = tf.log(tf.clip_by_value(policy, 1e-15, 1 - 1e-15))
        return -tf.reduce_mean(tf.reduce_sum(r * log_policy, axis=1))


class InstRegretStratAvg(object):
    '''Bad and doesn't make much sense.'''
    def policy_activation(self, pre_activations):
        return rrm.rm_policy(pre_activations)

    def loss(self, predictions, policy, cfv):
        r = tf.stop_gradient(
            rrm.rm_policy(cfv -
                          tf.reduce_sum(cfv * policy, axis=1, keepdims=True)))
        error = tf.square(r - predictions) / 2.0
        return tf.reduce_mean(tf.reduce_sum(error, axis=1))


class MetaRmp(object):
    def __init__(self, policies, *args, use_cumulative_values=False, **kwargs):
        super(MetaRmp, self).__init__(*args, **kwargs)
        self.policies = policies
        self.meta_qregrets = ResourceVariable(tf.zeros([len(policies), 1]),
                                              trainable=False)
        self.use_cumulative_values = use_cumulative_values

    def num_policies(self):
        return len(self.policies)

    def meta_policy(self):
        return rrm.rm_policy(self.meta_qregrets)

    def policy_activation(self, predictions):
        policies = tf.stack([policy(predictions) for policy in self.policies],
                            axis=-1)
        meta_policy = tf.reshape(self.meta_policy(),
                                 [1, 1, self.num_policies()])
        return tf.reduce_sum(policies * meta_policy, axis=-1)

    def __call__(self, inputs, rewards):
        return ContextualKofnGame(
            tf.squeeze(self.template.prob_ith_element_is_sampled), rewards,
            self.policy(inputs))

    def loss_and_grad(self, inputs, rewards):
        meta_policy = self.meta_policy()
        with tf.GradientTape() as tape:
            predictions = self.model(inputs)

            policies = tf.stack(
                [policy(predictions) for policy in self.policies], axis=-1)
            expanded_meta_policy = tf.reshape(self.meta_policy(),
                                              [1, 1, self.num_policies()])
            policy = tf.reduce_sum(policies * expanded_meta_policy, axis=-1)

            r = tf.stop_gradient(
                ContextualKofnGame(
                    tf.squeeze(self.template.prob_ith_element_is_sampled),
                    rewards, policy).cfv)
            loss_value = self.loss(predictions, policy, r)

        evs = tf.reduce_mean(tf.reduce_sum(tf.expand_dims(r, axis=-1) *
                                           policies,
                                           axis=1),
                             axis=0,
                             keepdims=True)
        inst_r = evs - tf.matmul(evs, meta_policy)
        grad = tape.gradient(loss_value, self.model.variables)
        return loss_value, (zip(grad, self.model.variables), inst_r)

    def apply(self, grad):
        self.optimizer.apply_gradients(grad[0])
        self.meta_qregrets.assign(tf.maximum(grad[1] + self.meta_qregrets,
                                             0.0))
        return self


class Rrm(MetaRmp):
    def __init__(self,
                 *args,
                 ignore_negative_regrets=False,
                 softmax_temperatures=[],
                 use_cumulative_values=False,
                 **kwargs):
        self.use_cumulative_values = use_cumulative_values

        def f(temp):
            def g(x):
                return tf.nn.softmax(x / self._adjusted_temperature(temp))

        policies = ([rrm.rm_policy] + list(map(f, softmax_temperatures)))
        super(Rrm, self).__init__(policies, *args, **kwargs)
        self.ignore_negative_regrets = ignore_negative_regrets

    def _adjusted_temperature(self, temp):
        return (
            temp /
            (
                tf.cast(tf.train.get_global_step(), tf.float32) + 1.0
                if self.use_cumulative_values else 1.0
            )
        )  # yapf:disable

    def loss(self, predictions, policy, cfv):
        return rrm_loss(predictions,
                        cfv,
                        ignore_negative_regrets=self.ignore_negative_regrets)


class SplitRrm(MetaRmp):
    '''Should be better than RRM in some cases but haven't seen it yet.'''
    def __init__(self,
                 *args,
                 softmax_temperatures=[],
                 use_cumulative_values=False,
                 **kwargs):
        def f(temp):
            def g(z):
                return tf.nn.softmax(z[:, :-1] /
                                     self._adjusted_temperature(temp))

        policies = ([lambda z: rrm.rm_policy(z[:, :-1] - z[:, -1:])] +
                    list(map(f), softmax_temperatures))
        super(SplitRrm, self).__init__(policies, *args, **kwargs)

    def regrets(self, inputs):
        z = self.model(inputs)
        return z[:, :-1] - z[:, -1:]

    def loss(self, predictions, policy, cfv):
        q, v = predictions[:, :-1], predictions[:, -1:]
        r = q - v

        pi_rm = rrm.rm_policy(r)

        q_diffs = tf.square(q - cfv)
        q_loss = tf.reduce_mean(tf.reduce_sum(q_diffs, axis=1)) / 2.0

        ev = tf.stop_gradient(tf.reduce_sum(cfv * pi_rm, axis=1,
                                            keepdims=True))

        v_loss = tf.reduce_mean(tf.square(v - ev)) / 2.0
        return q_loss + v_loss


class Rrmp(Rrm):
    '''With a tabular representation, this reduces to RM+.'''
    def loss(self, predictions, policy, cfv):
        pi = rrm.rm_policy(predictions)
        inst_r = cfv - rrm.utility(pi, cfv)
        inst_q = tf.stop_gradient(tf.maximum(inst_r, -tf.nn.relu(predictions)))
        return tf.reduce_mean(
            tf.reduce_sum(tf.square(predictions - inst_q), axis=1)) / 2.0
