import tensorflow as tf
import numpy as np
from driving_gridworld.road import Road
from driving_gridworld.car import Car
from driving_gridworld.obstacles import Bump, Pedestrian
from tf_kofn_robust_policy_optimization.discounted_mdp import \
    state_successor_policy_evaluation_op, \
    dual_state_value_policy_evaluation_op, \
    state_distribution
from research2018.tabular_cfr import TabularCfr


def new_road(headlight_range=3,
             allow_crashing=False,
             car=None,
             obstacles=None,
             allowed_obstacle_appearance_columns=None):
    return Road(
        headlight_range,
        Car(2, 0) if car is None else car,
        obstacles=([
            Bump(-1, -1, prob_of_appearing=0.5),
            Pedestrian(-1, -1, speed=1, prob_of_appearing=0.5)
        ] if obstacles is None else obstacles),
        allowed_obstacle_appearance_columns=(
            [{2}, {1}] if allowed_obstacle_appearance_columns is None else
            allowed_obstacle_appearance_columns),
        allow_crashing=allow_crashing)


def safety_info(root_probs, transitions, sa_safety_info, policy, discount=1.0):
    '''Assumes the first dimension is a batch dimension.'''
    state_safety_info = dual_state_value_policy_evaluation_op(
        transitions, policy, sa_safety_info, gamma=discount)

    if len(state_safety_info.shape) < 2:
        state_safety_info = tf.expand_dims(state_safety_info, 0)
    else:
        state_safety_info = tf.transpose(state_safety_info)

    root_probs = tf.convert_to_tensor(root_probs)
    if len(root_probs.shape) < 2:
        root_probs = tf.expand_dims(root_probs, 0)
    return tf.reduce_sum(root_probs * state_safety_info, axis=-1)


class KofnCfr(object):
    '''k-of-n mixin designed to override a `FixedParameterCfr` class.'''

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


class UrdcKofnTabularCfr(KofnCfr):
    '''
    k-of-n mixin specific to tabular uncertain reward discounted continuing
    MDPs designed to override a `FixedParameterCfr` class.
    '''

    @classmethod
    def from_num_states_and_actions(cls, num_states, num_actions, **kwargs):
        return cls(cfr=TabularCfr.zeros(num_states, num_actions), **kwargs)

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
