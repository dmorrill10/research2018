import tensorflow as tf
import numpy as np
from driving_gridworld.road import Road
from driving_gridworld.car import Car
from driving_gridworld.obstacles import Bump, Pedestrian
from driving_gridworld.rewards import TfUniformSituationalReward
from tf_kofn_robust_policy_optimization.discounted_mdp import \
    state_successor_policy_evaluation_op, \
    state_distribution
from tf_kofn_robust_policy_optimization.robust.uncertain_reward_discounted_continuing_kofn import \
    UncertainRewardDiscountedContinuingKofnGame, \
    UncertainRewardDiscountedContinuingKofnEvEnv
from research2018.tabular_cfr import FixedParameterAvgCodeCfr, TabularCfr


def new_road(headlight_range=3,
             allow_crashing=False,
             car=None,
             obstacles=None,
             allowed_obstacle_appearance_columns=None):
    return Road(
        headlight_range,
        Car(2, 0) if car is None else car,
        obstacles=([Bump(-1, -1), Pedestrian(-1, -1, speed=1)]
                   if obstacles is None else obstacles),
        allowed_obstacle_appearance_columns=(
            [{2}, {1}] if allowed_obstacle_appearance_columns is None else
            allowed_obstacle_appearance_columns),
        allow_crashing=allow_crashing)


def new_random_reward_function(stopping_reward=0,
                               positive_distribution=None,
                               critical_error_reward_bonus=-1000):
    if positive_distribution is None:
        positive_distribution = tf.distributions.Exponential(1.0)
    sampled = positive_distribution.sample(2)
    wc_non_critical_error_reward = stopping_reward - sampled[0]

    return TfUniformSituationalReward(
        wc_non_critical_error_reward=wc_non_critical_error_reward,
        stopping_reward=stopping_reward,
        reward_for_critical_error=(
            wc_non_critical_error_reward + critical_error_reward_bonus),
        max_unobstructed_progress_reward=sampled[1] + stopping_reward)


def safety_info(root_probs,
                transitions,
                sa_safety_info,
                policy,
                avg_threshold=1e-7,
                **kwargs):
    policy_weighted_safety_info = tf.reduce_sum(
        sa_safety_info * tf.expand_dims(policy, axis=-1), axis=1)

    root_probs = tf.squeeze(tf.convert_to_tensor(root_probs))
    num_states = root_probs.shape[0].value
    sd = tf.squeeze(
        state_distribution(
            state_successor_policy_evaluation_op(
                transitions,
                policy,
                threshold=avg_threshold * num_states,
                **kwargs), root_probs))

    return tf.reduce_sum(
        tf.expand_dims(sd, axis=-1) * policy_weighted_safety_info, axis=0)


def new_transitions_rewards_and_indices(road, reward_function=None, **kwargs):
    transitions, reward_function_dist, state_indices = road.tabulate(
        (new_random_reward_function()
         if reward_function is None else reward_function), **kwargs)
    return tf.stack(transitions), tf.stack(reward_function_dist), state_indices


def new_transitions_rewards_and_indices_on_cpu(*args, **kwargs):
    with tf.device('/cpu:0'):
        ret = new_transitions_rewards_and_indices(*args, **kwargs)
    return ret


class FixedRewardEnvFactory(object):
    def __init__(self, root_probs, transitions, rewards, **kwargs):
        self.root_probs = root_probs
        self.transitions = tf.convert_to_tensor(transitions)
        self.rewards = rewards
        self.kwargs = kwargs

    def num_states(self):
        return self.transitions.shape[0].value

    def num_actions(self):
        return self.transitions.shape[1].value

    def num_state_actions(self):
        return self.num_states() * self.num_actions()

    def sample_reward_function(self):
        return self.rewards

    def action_env(self,
                   kofn_opponent,
                   avg_threshold=0.1,
                   max_num_iterations=1000):
        return UncertainRewardDiscountedContinuingKofnGame.environment(
            kofn_opponent,
            self.root_probs,
            self.transitions,
            self.sample_reward_function,
            threshold=avg_threshold * self.num_state_actions(),
            max_num_iterations=max_num_iterations,
            **self.kwargs)

    def state_env(self,
                  kofn_opponent,
                  avg_threshold=0.1,
                  max_num_iterations=1000):
        return UncertainRewardDiscountedContinuingKofnEvEnv(
            kofn_opponent,
            self.root_probs,
            self.transitions,
            self.sample_reward_function,
            threshold=avg_threshold * self.num_states(),
            max_num_iterations=max_num_iterations,
            **self.kwargs)


class KofnCfr(FixedParameterAvgCodeCfr):
    def __init__(self, opponent, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.opponent = opponent

    def params(self, sess=None):
        if sess is None:
            return [self.opponent] + super().params()
        else:
            return [sess.run(self.opponent)] + super().params()

    def graph_save(self, name, sess):
        np.save('{}.params'.format(name), self.params(sess))
        self.cfr.graph_save('{}.cfr'.format(name), sess)
        return self


class UncertainRewardDiscountedContinuingKofnTabularCfr(KofnCfr):
    @classmethod
    def from_num_states_and_actions(cls, num_states, num_actions, **kwargs):
        return cls(cfr=TabularCfr.zeros(num_states, num_actions), **kwargs)

    def __init__(self, pe_threshold, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pe_threshold = pe_threshold

    def params(self, *args, **kwargs):
        return super().params(*args, **kwargs) + [self.pe_threshold]

    @property
    def policy(self):
        return self.cfr.policy

    def state_successor_rep(self,
                            transitions,
                            discount=1.0,
                            avg_threshold=1e-7,
                            max_num_iterations=1000):
        transitions = tf.convert_to_tensor(transitions)
        return state_successor_policy_evaluation_op(
            transitions,
            self.cfr.policy(),
            gamma=discount,
            threshold=avg_threshold * transitions.shape[0].value,
            max_num_iterations=max_num_iterations)

    def state_distribution(self, root_probs, transitions, **kwargs):
        return state_distribution(
            self.state_successor_rep(transitions, **kwargs), root_probs)

    def state_action_distribution(self, **kwargs):
        return (tf.expand_dims(self.state_distribution(**kwargs), axis=-1) *
                self.policy())


class UncertainRewardDiscountedContinuingKofnLearner(object):
    def __init__(self, cfr, new_train_env, new_test_env):
        self.cfr = cfr
        self.train_env = new_train_env(cfr.opponent, cfr.pe_threshold)
        self.test_env = new_test_env(cfr.opponent)

    @property
    def env_update(self):
        return self.train_env.update

    @property
    def policy(self):
        return self.cfr.policy

    @property
    def test_env_update(self):
        return self.test_env.update

    def cfr_update(self, *args, **kwargs):
        return self.cfr.update(self.train_env, *args, **kwargs)

    def test_ev(self):
        return tf.squeeze(self.test_env(self.policy()))
