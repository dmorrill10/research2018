import tensorflow as tf
import numpy as np
from tensorflow.python.ops.resource_variable_ops import ResourceVariable
import yaml

from tf_kofn_robust_policy_optimization.robust.kofn import \
    ContextualKofnGame, \
    DeterministicKofnGameTemplate
from tf_contextual_prediction_with_expert_advice import \
    rm_policy, \
    utility, \
    norm_exp
from tf_contextual_prediction_with_expert_advice.rrm import rrm_loss

from robust_offline_contextual_bandits.policy import \
    sorted_values_across_worlds
from simple_pytimer import AccumulatingTimer

import matplotlib.pyplot as plt


class KofnLearner(object):
    # TODO: Separate from optimizer
    @classmethod
    def load(cls, name, optimizer):
        with open('{}.yml'.format(name), "r") as yaml_file:
            data_string = yaml_file.read()

        # Keras saves models in a way that requires an unsafe load
        data = yaml.load(data_string)
        kofn_data = data.pop('kofn')
        template = DeterministicKofnGameTemplate(kofn_data['k'],
                                                 kofn_data['n'])

        model = tf.keras.models.model_from_yaml(data_string)
        model.load_weights('{}.h5'.format(name))
        return cls(template, model, optimizer, label=data.pop('label'))

    def __init__(self, template, model, optimizer, label=None):
        self.model = model
        self.template = template
        self.optimizer = optimizer
        self.label = label

    def save(self, name):
        model_yaml = (
            self.model.to_yaml()
            + "label: {}\n".format(self.label)
            + "kofn:\n{}".format(self.template.to_yml(indent=2))
        )  # yapf:disable

        with open('{}.yml'.format(name), "w") as yaml_file:
            yaml_file.write(model_yaml)
        self.model.save_weights('{}.h5'.format(name))

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
        r = norm_exp(kofn_utility, temp=1e-15)
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
        return norm_exp(pre_activations)

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
                    lambda temp: lambda x: norm_exp(x, self._adjusted_temperature(temp)),
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
                    lambda temp: lambda z: norm_exp(z[:, :-1], self._adjusted_temperature(temp)),
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
        inst_q = tf.stop_gradient(tf.maximum(inst_r, -tf.nn.relu(predictions)))
        return tf.reduce_mean(
            tf.reduce_sum(tf.square(predictions - inst_q), axis=1)) / 2.0

    def method_name(self):
        return 'RRM+-{}'.format(self.template.label())


class KofnTrainingData(object):
    @classmethod
    def empty(cls, num_learners):
        return cls([[] for _ in range(num_learners)],
                   [[] for _ in range(num_learners)], [])

    @classmethod
    def load(cls, name):
        with open('{}.losses_over_time.npy'.format(name), 'rb') as f:
            lot = np.load(f)
        with open('{}.evs_over_time.npy'.format(name), 'rb') as f:
            eot = np.load(f)
        with open('{}.checkpoint_iterations.npy'.format(name), 'rb') as f:
            ci = np.load(f)
        return cls(lot.tolist(), eot.tolist(), ci)

    @classmethod
    def combine(cls, *data):
        return cls(
            sum([d._losses_over_time for d in data], []),
            sum([d._evs_over_time for d in data], []),
            data[0].checkpoint_iterations)

    def __init__(self, losses_over_time, evs_over_time, checkpoint_iterations):
        assert len(losses_over_time) == len(evs_over_time)
        for i, lot in enumerate(losses_over_time):
            assert len(lot) == len(evs_over_time[i])

        self._losses_over_time = losses_over_time
        self._evs_over_time = evs_over_time
        self.checkpoint_iterations = checkpoint_iterations

    def __len__(self):
        return len(self._losses_over_time)

    def __getitem__(self, i):
        if i < 0:
            i = len(self) - i
        return self.__class__(self._losses_over_time[i:i + 1],
                              self._evs_over_time[i:i + 1],
                              self.checkpoint_iterations)

    def loss_measurement_is_missing(self):
        return len(self._losses_over_time[0]) < len(self.checkpoint_iterations)

    def save(self, name):
        with open('{}.losses_over_time.npy'.format(name), 'wb') as f:
            np.save(f, self.losses_over_time)
        with open('{}.evs_over_time.npy'.format(name), 'wb') as f:
            np.save(f, self.evs_over_time)
        with open('{}.checkpoint_iterations.npy'.format(name), 'wb') as f:
            np.save(f, self.checkpoint_iterations)
        return self

    @property
    def losses_over_time(self):
        return tf.stack(self._losses_over_time).numpy()

    @property
    def evs_over_time(self):
        return tf.stack(self._evs_over_time).numpy()

    def losses_now(self):
        return self.losses_over_time[:, -1]

    def evs_now(self):
        return self.evs_over_time[:, -1]


class KofnResults(object):
    @classmethod
    def combine(cls, *results):
        return cls(
            sum([r.learners for r in results]),
            KofnTrainingData.combine([r.data for r in results]))

    def __init__(self, learners, data):
        assert len(learners) == len(data)
        for i, learners_in_reality in enumerate(self.learners):
            assert len(learners_in_reality) == len(data[i])
        self.learners = learners
        self.data = data

    def lotor(self):
        '''reality x learner x time'''
        return np.array([d.losses_over_time for d in self.data])

    def eotor(self):
        '''reality x learner x time'''
        return np.array([d.evs_over_time for d in self.data])

    @property
    def checkpoint_iterations(self):
        return self.data[0].checkpoint_iterations

    @property
    def learner_names(self):
        return [str(learner) for learner in self.learners[0]]


class KofnTraining(object):
    def __init__(self,
                 trainer,
                 input_generator,
                 reward_sampling_timer,
                 num_iterations=1000,
                 num_ts_between_saving_checkpoints=1,
                 num_display_checkpoints=10):
        self.trainer = trainer
        self.input_generator = input_generator
        self.num_iterations = num_iterations
        self.num_ts_between_saving_checkpoints = num_ts_between_saving_checkpoints
        self.num_display_checkpoints = num_display_checkpoints
        self.reward_sampling_timer = reward_sampling_timer

        self.kofn_timer = AccumulatingTimer('k-of-n training')
        self._data = KofnTrainingData.empty(len(self.learners))

    def save_data(self, name):
        self._data.save(name)
        return self

    @property
    def losses_over_time(self):
        return self._data.losses_over_time

    @property
    def evs_over_time(self):
        return self._data.evs_over_time

    @property
    def checkpoint_iterations(self):
        return self._data.checkpoint_iterations

    @property
    def learners(self):
        return self.trainer.learners

    @property
    def t(self):
        return self.trainer.t

    def next_input(self):
        return next(self.input_generator())

    def checkpoint_is_missing(self):
        return (len(self.checkpoint_iterations) < 1
                or self.checkpoint_iterations[-1] < self.t)

    def is_last_iteration(self):
        return self.t % self.num_iterations == 0

    def is_checkpoint_iteration(self):
        return (self.t % self.num_ts_between_saving_checkpoints == 0
                or self.is_last_iteration())

    def is_display_iteration(self):
        return (self.t %
                (self.num_iterations // self.num_display_checkpoints) == 0
                or self.is_last_iteration())

    def plot_loss_curves(self, policy_alg_styles):
        _lot = self.losses_over_time
        for i, l in enumerate(_lot):
            t = self.learners[i]
            plt.plot(
                self.checkpoint_iterations,
                l,
                alpha=0.3,
                **policy_alg_styles[str(t)])
            plt.plot(
                self.checkpoint_iterations,
                l.cumsum() / np.arange(1, len(l) + 1),
                label=str(t),
                **policy_alg_styles[str(t)])  # yapf:disable
        plt.legend()
        plt.xlabel('iteration')
        plt.ylabel('MSE')

    def plot_ev_curves(self, policy_alg_styles, with_avg=False):
        _eot = self.evs_over_time
        for i, l in enumerate(_eot):
            t = self.learners[i]

            if with_avg:
                plt.plot(
                    self.checkpoint_iterations,
                    l,
                    alpha=0.2,
                    **policy_alg_styles[str(t)])
                plt.plot(
                    self.checkpoint_iterations,
                    l.cumsum() / np.arange(1, len(l) + 1),
                    label=str(t),
                    **policy_alg_styles[str(t)])  # yapf:disable
            else:
                plt.plot(
                    self.checkpoint_iterations,
                    l,
                    label=str(t),
                    **policy_alg_styles[str(t)])
        plt.legend()
        plt.xlabel('iteration')
        plt.ylabel('training EV')

    def reward_across_inputs(self, rewards):
        return [self.utility(learner, rewards) for learner in self.learners]

    def sorted_values_across_worlds(self, rewards):
        return [
            sorted_values_across_worlds(
                learner.policy(self.next_input()), rewards)
            for learner in self.learners
        ]

    def utility(self, learner, rewards):
        return utility(learner.policy(self.next_input()), rewards)

    def value_slopes_and_biases(self, arms_with_contexts):
        test_rewards = [
            arms_with_contexts.with_function_outside_plateaus(lambda _: i)
            .combined_raw_y() for i in range(2)
        ]
        slopes_and_biases = []
        for learner in self.learners:
            bias = tf.reduce_mean(self.utility(learner, test_rewards[0]))
            slope = tf.reduce_mean(self.utility(learner,
                                                test_rewards[1])) - bias
            slopes_and_biases.append((slope.numpy(), bias.numpy()))
        return slopes_and_biases

    def loss_measurement_is_missing(self):
        return self._data.loss_measurement_is_missing()

    def losses_now(self):
        return self._data.losses_now()

    def evs_now(self):
        return self._data.evs_now()

    def evaluate(self):
        return self.trainer.evaluate(self.next_input())

    def run(self):
        if self.checkpoint_is_missing():
            evs = self.evaluate()

            for i in range(len(self.learners)):
                self._data._evs_over_time[i].append(evs[i])
            self._data.checkpoint_iterations.append(self.t)

            print('{}: ev: {}'.format(self.t, evs))

        with self.kofn_timer:
            for iteration in range(self.num_iterations):
                for phi in self.input_generator():
                    losses = self.trainer.step(phi)
                    if self.loss_measurement_is_missing():
                        for i in range(len(self.learners)):
                            self._data._losses_over_time[i].append(losses[i])

                    if self.is_checkpoint_iteration():
                        evs = self.evaluate()

                        for i in range(len(self.learners)):
                            self._data._evs_over_time[i].append(evs[i])
                            self._data._losses_over_time[i].append(losses[i])
                        self._data.checkpoint_iterations.append(self.t)

                    if self.is_display_iteration():
                        progress = '{}/{}'.format(iteration + 1,
                                                  self.num_iterations)
                        print(
                            '{}: t: {}\n{}  loss: {}\n{}  ev: {}'.format(
                                progress,
                                self.t,
                                ' ' * len(progress),
                                self.losses_now(),
                                ' ' * len(progress),
                                self.evs_now()
                            )
                        )  # yapf:disable
                        print(self.reward_sampling_timer)

                        self.kofn_timer.mark()
                        print(self.kofn_timer)
