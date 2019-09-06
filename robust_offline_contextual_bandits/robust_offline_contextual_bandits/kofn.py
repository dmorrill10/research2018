import tensorflow as tf
import numpy as np

from research2018 import rrm

from robust_offline_contextual_bandits.policy import \
    sorted_values_across_worlds
from simple_pytimer import AccumulatingTimer

from robust_offline_contextual_bandits.representations import \
    RepresentationWithFixedInputs

from robust_offline_contextual_bandits import cache

import matplotlib.pyplot as plt


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
        return cls(sum([d._losses_over_time for d in data], []),
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


class KofnTraining(object):
    def __init__(self,
                 trainer,
                 input_generator,
                 reward_sampling_timer=None,
                 kofn_timer=None,
                 num_iterations=1000,
                 num_ts_between_saving_checkpoints=1,
                 num_display_checkpoints=10):
        self.trainer = trainer
        self.input_generator = input_generator
        self.num_iterations = num_iterations
        self.num_ts_between_saving_checkpoints = (
            num_ts_between_saving_checkpoints)
        self.num_display_checkpoints = num_display_checkpoints

        if reward_sampling_timer is None:
            reward_sampling_timer = AccumulatingTimer('reward sampling')
        self.reward_sampling_timer = reward_sampling_timer

        if kofn_timer is None:
            kofn_timer = AccumulatingTimer('k-of-n training')
        self.kofn_timer = kofn_timer

        self._data = KofnTrainingData.empty(len(self.learners))

    def save_data(self, name):
        self._data.save(name)
        return self

    @property
    def data(self):
        return self._data

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
            plt.plot(self.checkpoint_iterations,
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
                plt.plot(self.checkpoint_iterations,
                         l,
                         alpha=0.2,
                         **policy_alg_styles[str(t)])
                plt.plot(
                    self.checkpoint_iterations,
                    l.cumsum() / np.arange(1, len(l) + 1),
                    label=str(t),
                    **policy_alg_styles[str(t)])  # yapf:disable
            else:
                plt.plot(self.checkpoint_iterations,
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
            sorted_values_across_worlds(learner.policy(self.next_input()),
                                        rewards) for learner in self.learners
        ]

    def utility(self, learner, rewards):
        return rrm.utility(learner.policy(self.next_input()), rewards)

    def value_slopes_and_biases(self, arms_with_contexts):
        test_rewards = [
            arms_with_contexts.with_function_outside_plateaus(
                lambda _: i).combined_raw_y() for i in range(2)
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
        # if self.checkpoint_is_missing():
        #     evs = self.evaluate()
        #
        #     for i in range(len(self.learners)):
        #         self._data._evs_over_time[i].append(evs[i])
        #     self._data.checkpoint_iterations.append(self.t)
        #
        #     print('{}: ev: {}'.format(self.t, evs))

        with self.kofn_timer:
            for iteration in range(self.num_iterations):
                for phi in self.input_generator():
                    losses, evs = self.trainer.step(phi)

                    # TODO This is unneccessary
                    for i in range(len(self.learners)):
                        self._data._evs_over_time[i].append(evs[i])
                        self._data._losses_over_time[i].append(losses[i])

                    # if self.loss_measurement_is_missing():
                    #     for i in range(len(self.learners)):
                    #         self._data._losses_over_time[i].append(losses[i])

                    if self.is_checkpoint_iteration():
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


class KofnTrainingResults(object):
    @classmethod
    def from_competitor(cls, competitor, training_data):
        return cls(competitor.rep, competitor.policy_model, training_data)

    @classmethod
    def load(cls, name, policy_model):
        return cls(RepresentationWithFixedInputs.load('{}.rep'.format(name)),
                   policy_model.load('{}.policy_model'.format(name)),
                   KofnTrainingData.load('{}.training_data'.format(name)))

    def __init__(self, rep, policy_model, training_data):
        self.rep = rep
        self.policy_model = policy_model
        self.training_data = training_data

    def save(self, name):
        self.rep.save('{}.rep'.format(name))
        self.policy_model.save('{}.policy_model'.format(name))
        self.training_data.save('{}.training_data'.format(name))
        return self

    @cache
    def policy(self):
        return self.policy_model(self.rep.phi)
