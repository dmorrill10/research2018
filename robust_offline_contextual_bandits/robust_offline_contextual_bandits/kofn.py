import tensorflow as tf
from tensorflow.python.ops.resource_variable_ops import ResourceVariable
from tf_kofn_robust_policy_optimization.robust.kofn import ContextualKofnGame
from tf_contextual_prediction_with_expert_advice import rm_policy, utility
from tf_contextual_prediction_with_expert_advice.rrm import rrm_loss
from robust_offline_contextual_bandits.policy import softmax


class KofnLearner(object):
    def __init__(self, template, model, optimizer, label=None):
        self.model = model
        self.template = template
        self.optimizer = optimizer
        self.label = label

    def policy(self, inputs):
        return self.policy_activation(self.model(inputs))

    def policy_activation(self, pre_activations):
        raise RuntimeError('Unimplemented')

    def __call__(self, inputs, rewards):
        return ContextualKofnGame(
            tf.squeeze(self.template.prob_ith_element_is_sampled), rewards,
            self.policy(inputs))

    def loss(self, predictions, policy, kofn_utility):
        raise RuntimeError('Unimplemented')

    def loss_and_grad(self, inputs, rewards):
        with tf.GradientTape() as tape:
            pre_activations = self.model(inputs)
            policy = self.policy_activation(pre_activations)
            r = tf.stop_gradient(
                ContextualKofnGame(
                    tf.squeeze(self.template.prob_ith_element_is_sampled),
                    rewards, policy).kofn_utility)
            loss_value = self.loss(pre_activations, policy, r)
        return (loss_value,
                zip(
                    tape.gradient(loss_value, self.model.variables),
                    self.model.variables))

    def apply(self, grad):
        return self.optimizer.apply_gradients(grad)

    def _name_suffix(self):
        return '' if self.label is None else '-{}'.format(self.label)

    def __str__(self):
        return '{}{}'.format(self.method_name(), self._name_suffix())


class KofnRewardAugmentedMlLearner(KofnLearner):
    def policy_activation(self, pre_activations):
        return pre_activations

    def loss(self, predictions, policy, kofn_utility):
        r = softmax(kofn_utility, temp=1e-15)
        log_policy = tf.log(tf.clip_by_value(policy, 1e-15, 1 - 1e-15))
        return -tf.reduce_mean(r * log_policy)

    def method_name(self):
        return 'RAML-{}'.format(self.template.label())


class KofnPolicyGradientLearner(KofnLearner):
    def policy_activation(self, pre_activations):
        return pre_activations

    def loss(self, predictions, policy, kofn_utility):
        return -tf.reduce_mean(tf.reduce_sum(kofn_utility * policy, axis=1))

    def method_name(self):
        return 'PG-{}'.format(self.template.label())


class KofnLinearLossLearner(KofnLearner):
    '''Can't get this to work better than just predicting values/RRM and maybe it shouldn't.'''

    def policy_activation(self, pre_activations):
        return softmax(pre_activations)

    def loss(self, predictions, policy, kofn_utility):
        return (-tf.reduce_mean(kofn_utility * predictions) + tf.reduce_mean(
            tf.reduce_logsumexp(tf.abs(predictions), axis=1)))
        # max_pseudo_regret = tf.reduce_mean(
        #     tf.reduce_max(
        #         tf.maximum(
        #             0.0,
        #             r - tf.reduce_sum(r * predictions, axis=1, keepdims=True)),
        #         axis=1))
        # return (
        #     max_pseudo_regret
        #     + tf.reduce_mean(tf.reduce_logsumexp(predictions, axis=1)))

    def method_name(self):
        return 'LL-{}'.format(self.template.label())


class KofnInstRegretStratMatchingLearner(KofnLearner):
    '''Bad even though it makes some sense.'''

    def policy_activation(self, pre_activations):
        return pre_activations

    def loss(self, predictions, policy, kofn_utility):
        r = tf.stop_gradient(
            rm_policy(kofn_utility - tf.reduce_sum(
                kofn_utility * policy, axis=1, keepdims=True)))
        log_policy = tf.log(tf.clip_by_value(policy, 1e-15, 1 - 1e-15))
        return -tf.reduce_mean(tf.reduce_sum(r * log_policy, axis=1))

    def method_name(self):
        return 'IRSM-{}'.format(self.template.label())


class KofnInstRegretStratAvgLearner(KofnLearner):
    '''Bad and doesn't make much sense.'''

    def policy_activation(self, pre_activations):
        return rm_policy(pre_activations)

    def loss(self, predictions, policy, kofn_utility):
        r = tf.stop_gradient(
            rm_policy(kofn_utility - tf.reduce_sum(
                kofn_utility * policy, axis=1, keepdims=True)))
        error = tf.square(r - predictions) / 2.0
        return tf.reduce_mean(tf.reduce_sum(error, axis=1))

    def method_name(self):
        return 'IRSA-{}'.format(self.template.label())


class KofnMetaRmpLearner(KofnLearner):
    def __init__(self, policies, *args, use_cumulative_values=False, **kwargs):
        super(KofnMetaRmpLearner, self).__init__(*args, **kwargs)
        self.policies = policies
        self.meta_qregrets = ResourceVariable(
            tf.zeros([len(policies), 1]), trainable=False)
        self.use_cumulative_values = use_cumulative_values

    def num_policies(self):
        return len(self.policies)

    def meta_policy(self):
        return rm_policy(self.meta_qregrets)

    def policy_activation(self, predictions):
        policies = tf.stack(
            [policy(predictions) for policy in self.policies], axis=-1)
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
                    rewards, policy).kofn_utility)
            loss_value = self.loss(predictions, policy, r)

        evs = tf.reduce_mean(
            tf.reduce_sum(tf.expand_dims(r, axis=-1) * policies, axis=1),
            axis=0,
            keepdims=True)
        inst_r = evs - tf.matmul(evs, meta_policy)
        grad = tape.gradient(loss_value, self.model.variables)
        return loss_value, (zip(grad, self.model.variables), inst_r)

    def apply(self, grad):
        self.optimizer.apply_gradients(grad[0])
        self.meta_qregrets.assign(
            tf.maximum(grad[1] + self.meta_qregrets, 0.0))
        return self


class KofnRrmLearner(KofnMetaRmpLearner):
    def __init__(self,
                 *args,
                 ignore_negative_regrets=False,
                 softmax_temperatures=[],
                 use_cumulative_values=False,
                 **kwargs):
        self.use_cumulative_values = use_cumulative_values
        policies = (
            [rm_policy]
            + list(
                map(
                    lambda temp: lambda x: softmax(x, self._adjusted_temperature(temp)),
                    softmax_temperatures
                )
            )
        )  # yapf:disable
        super(KofnRrmLearner, self).__init__(policies, *args, **kwargs)
        self.ignore_negative_regrets = ignore_negative_regrets

    def _adjusted_temperature(self, temp):
        return (
            temp /
            (
                tf.cast(tf.train.get_global_step(), tf.float32) + 1.0
                if self.use_cumulative_values else 1.0
            )
        )  # yapf:disable

    def loss(self, predictions, policy, kofn_utility):
        return rrm_loss(
            predictions,
            kofn_utility,
            ignore_negative_regrets=self.ignore_negative_regrets)

    def method_name(self):
        return 'RRM-{}'.format(self.template.label())


class KofnSplitRrmLearner(KofnMetaRmpLearner):
    '''Should be better than RRM in some cases but haven't seen it yet.'''

    def __init__(self,
                 *args,
                 softmax_temperatures=[],
                 use_cumulative_values=False,
                 **kwargs):

        policies = (
            [lambda z: rm_policy(z[:, :-1] - z[:, -1:])]
            + list(
                map(
                    lambda temp: lambda z: softmax(z[:, :-1], self._adjusted_temperature(temp)),
                    softmax_temperatures
                )
            )
        )  # yapf:disable
        super(KofnSplitRrmLearner, self).__init__(policies, *args, **kwargs)

    def regrets(self, inputs):
        z = self.model(inputs)
        return z[:, :-1] - z[:, -1:]

    def loss(self, predictions, policy, kofn_utility):
        q, v = predictions[:, :-1], predictions[:, -1:]
        r = q - v

        pi_rm = rm_policy(r)

        q_diffs = tf.square(q - kofn_utility)
        q_loss = tf.reduce_mean(tf.reduce_sum(q_diffs, axis=1)) / 2.0

        ev = tf.stop_gradient(
            tf.reduce_sum(kofn_utility * pi_rm, axis=1, keepdims=True))

        v_loss = tf.reduce_mean(tf.square(v - ev)) / 2.0
        return q_loss + v_loss

    def method_name(self):
        return 'SRRM-{}'.format(self.template.label())


class KofnRrmpLearner(KofnRrmLearner):
    '''With a tabular representation, this reduces to RM+.'''

    def loss(self, predictions, policy, kofn_utility):
        pi = rm_policy(predictions)
        inst_r = kofn_utility - utility(pi, kofn_utility)
        diff = predictions - tf.stop_gradient(tf.minimum(inst_r, predictions))
        return tf.reduce_mean(tf.reduce_sum(tf.square(diff), axis=1))

    def method_name(self):
        return 'RRM+-{}'.format(self.template.label())


class KofnIterator(object):
    def __init__(self, trainer, input_generator):
        self._trainer = trainer
        self._input_generator = input_generator

    def __iter__(self):
        return self

    def __next__(self):
        return self._trainer.step(next(self._input_generator))


class KofnTrainer(object):
    def __init__(self, reward_generator, learners):
        self._t = tf.train.get_or_create_global_step()
        self._t.assign(0)
        self.reward_generator = reward_generator
        self.learners = learners

    @property
    def t(self):
        return int(self._t.numpy())

    def start(self, input_generator):
        return KofnIterator(self, input_generator)

    def step(self, inputs):
        reward = self.reward_generator(inputs)
        losses = []
        for i in range(len(self.learners)):
            learner = self.learners[i]
            loss, grad = learner.loss_and_grad(inputs, reward)
            losses.append(loss)
            learner.apply(grad)
        self._t.assign_add(1)
        return losses

    def evaluate(self, inputs, test_rewards):
        evs = []
        test_evs = []
        _r = self.reward_generator(inputs)
        for learner in self.learners:
            game = learner(inputs, _r)

            evs.append(game.root_ev.numpy())
            test_evs.append(
                tf.reduce_mean(utility(learner.policy(inputs),
                                       test_rewards)).numpy())
        return evs, test_evs
