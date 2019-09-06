import tensorflow as tf
import numpy as np

from driving_gridworld.road import Road
from driving_gridworld.car import Car
from driving_gridworld.obstacles import Bump, Pedestrian
from driving_gridworld.gridworld import DrivingGridworld
from driving_gridworld.rewards import \
    DebrisPerceptionReward, \
    fixed_ditch_bonus, \
    critical_reward_for_fixed_ditch_bonus

from tf_kofn_robust_policy_optimization.discounted_mdp import \
    state_successor_policy_evaluation_op, \
    dual_state_value_policy_evaluation_op, \
    state_distribution, \
    generalized_policy_iteration_op
from tf_kofn_robust_policy_optimization.robust.kofn import \
    KofnEvsAndWeights, \
    kofn_action_values
from tf_kofn_robust_policy_optimization.robust import \
    deterministic_kofn_weights

from research2018.tabular_cfr import TabularCfrCurrent
from research2018.kofn import KofnCfr
from research2018.fixed_parameter_cfr import FixedParameterCfr
from research2018.online_learner import OnlineLearner

from robust_offline_contextual_bandits import cache
from research2018.data import load_list
from robust_offline_contextual_bandits.plotting import \
    tableu20_color_table, \
    line_style_table
from robust_offline_contextual_bandits.tf_np import reset_random

import matplotlib.pyplot as plt

from simple_pytimer import AccumulatingTimer, Timer


def new_road(headlight_range=2):
    return Road(
        headlight_range,
        Car(2, 0),
        obstacles=[
            Bump(-1, -1, prob_of_appearing=0.1),
            Pedestrian(-1, -1, speed=1, prob_of_appearing=0.1)
        ],
        allowed_obstacle_appearance_columns=[{2}, {1}],
        allow_crashing=True)


def safety_info(root_probs,
                transitions,
                sa_safety_info,
                policy,
                discount=0.99,
                normalize=True):
    '''Assumes the first dimension is a batch dimension.'''
    state_safety_info = dual_state_value_policy_evaluation_op(
        transitions, policy, sa_safety_info, gamma=discount)

    if len(state_safety_info.shape) < 2:
        state_safety_info = tf.expand_dims(state_safety_info, 0)

    root_probs = tf.convert_to_tensor(root_probs)
    if len(root_probs.shape) < 2:
        root_probs = tf.expand_dims(root_probs, 0)

    discount = tf.convert_to_tensor(discount)
    discount = (tf.expand_dims(discount, 0)
                if len(discount.shape) == 1 else tf.transpose(discount))

    if normalize:
        state_safety_info = (1.0 - discount) * state_safety_info
    return tf.reduce_sum(root_probs * state_safety_info, axis=-1)


class UrdcKofnTabularCfr(KofnCfr):
    '''
    k-of-n mixin specific to tabular uncertain reward discounted continuing
    MDPs designed to override a `FixedParameterCfr` class.
    '''

    @classmethod
    def from_num_states_and_actions(cls,
                                    num_states,
                                    num_actions,
                                    cfr_cls=TabularCfrCurrent,
                                    **kwargs):
        return cls(cfr=cfr_cls.zeros(num_states, num_actions), **kwargs)

    @classmethod
    def train_env(cls, root_probs, transitions, reward_dataset, discount,
                  kofn_opponent):
        num_samples = reward_dataset.shape[0]
        n = reward_dataset.shape[1]
        transitions = tf.convert_to_tensor(transitions)
        num_states = transitions.shape[0]
        num_actions = transitions.shape[1]
        r = tf.reshape(reward_dataset,
                       [num_samples * n, num_states, num_actions])

        def env(policy):
            '''sample * world X state'''
            v = dual_state_value_policy_evaluation_op(
                transitions, policy, r, gamma=discount)
            q = tf.reshape(
                r + discount * tf.tensordot(v, transitions, axes=[-1, -1]),
                [num_samples, n, num_states, num_actions])
            '''sample X state X action X world'''
            q = tf.transpose(q, [0, 2, 3, 1])
            v = tf.transpose(
                tf.reshape(v, [num_samples, n, num_states]), [0, 2, 1])

            kofn_q = [
                kofn_action_values(
                    q[sample_idx],
                    KofnEvsAndWeights(
                        v[sample_idx],
                        kofn_opponent,
                        context_weights=root_probs).world_weights)
                for sample_idx in range(num_samples)
            ]
            return (1 - discount) * tf.reduce_mean(
                tf.stack(kofn_q, -1), axis=-1)

        return env

    @classmethod
    def test_env(cls, root_probs, transitions, reward_dataset, discount,
                 kofn_opponent):
        num_samples = reward_dataset.shape[0]
        n = reward_dataset.shape[1]
        transitions = tf.convert_to_tensor(transitions)
        num_states = transitions.shape[0]
        num_actions = transitions.shape[1]
        r = tf.reshape(reward_dataset,
                       [num_samples * n, num_states, num_actions])

        def env(policy):
            '''sample * world X state'''
            v = dual_state_value_policy_evaluation_op(
                transitions, policy, r, gamma=discount)
            v = tf.transpose(
                tf.reshape(v, [num_samples, n, num_states]), [0, 2, 1])
            kofn_ev = [
                KofnEvsAndWeights(
                    v[sample_idx], kofn_opponent,
                    context_weights=root_probs).ev
                for sample_idx in range(num_samples)
            ]
            return (1 - discount) * tf.reduce_mean(
                tf.stack(kofn_ev, -1), axis=-1)

        return env

    @property
    def policy(self):
        return self.cfr.policy

    def state_successor_rep(self, transitions, discount=1.0):
        transitions = tf.convert_to_tensor(transitions)
        return state_successor_policy_evaluation_op(
            transitions, self.policy(), gamma=discount)

    def state_distribution(self, root_probs, transitions, **kwargs):
        return state_distribution(
            self.state_successor_rep(transitions, **kwargs), root_probs)

    def state_action_distribution(self, **kwargs):
        return (tf.expand_dims(self.state_distribution(**kwargs), axis=-1) *
                self.policy())


class TabularRoad(object):
    @classmethod
    def tabulate(cls,
                 headlight_range=2,
                 num_samples_per_cfr_iter=10,
                 n=100,
                 loc=0,
                 precisions=[None],
                 discount=0.99,
                 progress_bonus=1.0,
                 ditch_bonus_multiplier=10.0,
                 normalize_rewards=False,
                 critical_error_reward=-1000.0,
                 print_every=100):
        speed_limit = new_road(headlight_range=headlight_range).speed_limit()
        game = DrivingGridworld(
            lambda: new_road(headlight_range=headlight_range))
        num_reward_functions = n * num_samples_per_cfr_iter
        wc_ncer = fixed_ditch_bonus(
            progress_bonus, multiplier=ditch_bonus_multiplier)

        if critical_error_reward is None:
            critical_error_reward = critical_reward_for_fixed_ditch_bonus(
                progress_bonus, speed_limit, discount)

        tf.logging.info('progress_bonus: {}, wc_ncer: {}, cer: {}'.format(
            progress_bonus, wc_ncer, critical_error_reward))

        reward_datasets = []
        for precision in precisions:
            random_reward_function = DebrisPerceptionReward(
                stopping_reward=tf.zeros([num_reward_functions]),
                wc_non_critical_error_reward=tf.fill([num_reward_functions],
                                                     wc_ncer),
                bc_unobstructed_progress_reward=tf.fill([num_reward_functions],
                                                        progress_bonus),
                num_samples=num_reward_functions,
                critical_error_reward=tf.fill([num_reward_functions],
                                              critical_error_reward),
                use_slow_collision_as_offroad_base=False,
                loc=loc,
                precision=precision)

            transitions, rfd_list, state_indices = game.road.tabulate(
                random_reward_function, print_every=print_every)

            r = tf.reshape(
                tf.transpose(tf.stack(rfd_list), [2, 0, 1]), [
                    num_samples_per_cfr_iter, n,
                    len(state_indices),
                    len(rfd_list[0])
                ])
            if normalize_rewards:
                r = r / tf.reduce_max(tf.abs(r), axis=(2, 3), keepdims=True)
            reward_datasets.append(r)

        transitions = tf.stack(transitions)
        root_probs = tf.one_hot(
            state_indices[game.road.copy().to_key()], depth=len(state_indices))
        return [
            cls(game, root_probs, transitions, r, discount, state_indices)
            for r in reward_datasets
        ]

    def __init__(self, game, root_probs, transitions, reward_dataset, discount,
                 state_indices):
        self.game = game
        self.root_probs = tf.convert_to_tensor(root_probs)
        self.transitions = tf.convert_to_tensor(transitions)
        self.reward_dataset = tf.convert_to_tensor(reward_dataset)
        self.state_indices = state_indices
        self.discount = discount

    @property
    def num_reps(self):
        return self.reward_dataset.shape[0]

    @property
    def num_worlds(self):
        return self.reward_dataset.shape[1]

    @property
    def num_states(self):
        return len(self.state_indices)

    @property
    def num_actions(self):
        return self.transitions.shape[1]

    @property
    def num_state_actions(self):
        return self.num_states * self.num_actions

    @cache
    def sa_safety_info(self):
        sasp_safety_info, _si = self.game.road.safety_information()
        for k, v in _si.items():
            assert self.state_indices[k] == v
        sasp_safety_info = tf.stack(sasp_safety_info)

        sa_safety_info = tf.transpose(
            tf.reduce_sum(
                sasp_safety_info * tf.expand_dims(self.transitions, axis=-1),
                axis=2), [2, 0, 1])
        return sa_safety_info

    @cache
    def discount_vector(self):
        terminal_states = tf.greater(self.sa_safety_info[0, :, -1], 0)
        discount = tf.expand_dims(
            tf.where(terminal_states, tf.zeros([self.num_states]),
                     tf.fill([self.num_states], self.discount)), -1)

        num_terminal_states = tf.reduce_sum(
            tf.cast(tf.greater(self.sa_safety_info[0, :, -1], 0),
                    tf.float32)).eval()
        assert num_terminal_states == 1
        return discount


class DgKofnTabularCfr(UrdcKofnTabularCfr, FixedParameterCfr):
    pass


class DgKofnTrainingResults(object):
    def __init__(self, reality_idx):
        self.reality_idx = reality_idx

    @load_list
    def load_learners(self, ks):
        self.learners = [
            DgKofnTabularCfr.load(
                'learner.{}.{}'.format(self.reality_idx, i),
                cfr_cls=TabularCfrCurrent) for i in range(len(ks))
        ]
        for learner in self.learners:
            tf.get_default_session().run(learner.cfr.regrets.initializer)
        return self.learners

    def load_eot(self):
        self.eot = np.load('eot.{}.npy'.format(self.reality_idx))
        return self.eot

    def load_ckpts(self):
        self.ckpts = np.load('ckpts.{}.npy'.format(self.reality_idx))
        return self.ckpts

    def load(self, ks):
        self.load_learners(ks)
        self.load_eot()
        self.load_ckpts()
        return self

    def assign(self, learners, eot, ckpts):
        self.learners = learners
        self.eot = eot
        self.ckpts = ckpts
        return self

    def save_learners(self):
        for i, learner in enumerate(self.learners):
            learner.graph_save('learner.{}.{}'.format(self.reality_idx, i),
                               tf.get_default_session())
        return self

    def save_eot(self):
        np.save('eot.{}'.format(self.reality_idx), self.eot)
        return self

    def save_ckpts(self):
        np.save('ckpts.{}'.format(self.reality_idx), self.ckpts)
        return self

    def save(self):
        self.save_learners()
        self.save_eot()
        self.save_ckpts()
        return self


class DgRealityExperiment(object):
    def __init__(self, reality_idx, road, ks):
        self.reality_idx = reality_idx
        self.road = road
        self.training_results = DgKofnTrainingResults(self.reality_idx)
        self.ks = ks

    @property
    def n(self):
        return self.road.num_worlds

    @cache
    def kofn_opponents(self):
        return [deterministic_kofn_weights(k, self.n) for k in self.ks]

    @property
    def discount_vector(self):
        return self.road.discount_vector

    @property
    def learners(self):
        return self.training_results.learners

    @property
    def eot_np(self):
        return self.training_results.eot

    @property
    def checkpoint_iterations(self):
        return self.training_results.ckpts

    @property
    def root_probs(self):
        return self.road.root_probs

    @property
    def transitions(self):
        return self.road.transitions

    @property
    def sa_safety_info(self):
        return self.road.sa_safety_info

    @property
    def num_states(self):
        return self.road.num_states

    @property
    def num_actions(self):
        return self.road.num_actions

    @property
    def num_state_actions(self):
        return self.road.num_state_actions

    @property
    def avg_crashes(self):
        crashes = self.safety_info[:, 0]
        assert np.all(crashes == 0)
        return crashes

    @property
    def avg_collisions(self):
        collisions = self.safety_info[:, 1]
        assert np.all(collisions == 0)
        return collisions

    @property
    def avg_debris(self):
        debris = self.safety_info[:, 2] / 0.5
        assert 0 <= max(debris) < 1
        return debris

    @property
    def avg_ditch(self):
        ditch = self.safety_info[:, 3] / 0.5
        assert 0 <= max(ditch) <= 1
        return ditch

    @property
    def avg_speeds(self):
        return self.safety_info[:, 4]

    @property
    def avg_progress(self):
        return self.safety_info[:, 5]

    @property
    def avg_lane_changes(self):
        return self.safety_info[:, 6]

    @property
    def avg_unsafe(self):
        unsafe = (self.avg_debris + self.avg_ditch) / 2.0
        assert max(unsafe) <= 1
        return unsafe

    @property
    def reward_dataset(self):
        return self.road.reward_dataset

    def policy_safety_info(self, policy):
        return safety_info(
            self.root_probs,
            self.transitions,
            self.sa_safety_info,
            policy,
            self.discount_vector,
            normalize=True)

    @cache
    def safety_info(self):
        return tf.convert_to_tensor([
            self.policy_safety_info(learner.policy())
            for learner in self.learners
        ]).eval()

    def train_learners(self,
                       num_iterations=5000,
                       use_plus=True,
                       print_every=10):
        eot = []
        checkpoint_iterations = []

        reset_random()

        timer = Timer('create learners')

        with timer:
            learners = [
                OnlineLearner(
                    DgKofnTabularCfr.from_num_states_and_actions(
                        self.num_states,
                        self.num_actions,
                        opponent=opponent,
                        use_plus=use_plus,
                        cfr_cls=TabularCfrCurrent),
                    DgKofnTabularCfr.train_env(
                        self.root_probs, self.transitions, self.reward_dataset,
                        self.discount_vector, opponent))
                for opponent in self.kofn_opponents
            ]
        print(timer)

        timer = Timer('create EV and update nodes')
        with timer:
            updates = []
            evs = []
            init = []
            for learner in learners:
                my_ev, my_update = learner.update()
                evs.append(tf.einsum('s,sa->', self.root_probs, my_ev))
                updates.append(my_update)
                init.append(learner.learner.cfr.regrets.initializer)
        print(timer)

        timer = Timer('init')
        with timer:
            tf.get_default_session().run(init)
        print(timer)

        timer = AccumulatingTimer('training')

        for t in range(1, num_iterations + 1):
            with timer:
                evs_np = tf.get_default_session().run([evs, updates])[0]
                eot.append(evs_np)
                checkpoint_iterations.append(t)
                if t == 1 or t % print_every == 0:
                    print('{}: {}'.format(t, evs_np))
                    print('{}: {}'.format(t, str(timer)))
        return [l.learner
                for l in learners], np.array(eot), checkpoint_iterations

    def train_and_save(self,
                       use_plus=True,
                       num_iterations=5000,
                       print_every=10):
        try:
            self.training_results.load(self.ks)
        except:
            self.training_results.assign(*self.train_learners(
                num_iterations=num_iterations,
                use_plus=use_plus,
                print_every=print_every)).save()
        return self

    def plot_convergence(self):
        colors = tableu20_color_table()
        colors = [next(colors) for _ in self.learners]

        line_styles = line_style_table()
        line_styles = [next(line_styles) for _ in self.learners]

        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches((10, 5))
        for ki, k in enumerate(self.ks):
            ax.plot(
                self.checkpoint_iterations,
                self.eot_np[:, ki],
                label='{}-of-{}'.format(k, self.n),
                color=colors[ki],
                ls=line_styles[ki])
        ax.set_xlabel('iteration')
        ax.set_ylabel('EV')
        plt.legend()
        return fig, ax

    def random_opt_policy(self, **kwargs):
        return generalized_policy_iteration_op(
            self.transitions,
            self.reward_dataset[0, 0],
            gamma=self.discount_vector,
            **kwargs)
