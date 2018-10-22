import tensorflow as tf
from driving_gridworld.road import Road
from driving_gridworld.car import Car
from driving_gridworld.obstacles import Bump, Pedestrian
from driving_gridworld.rewards import TfUniformSituationalReward
from tf_kofn_robust_policy_optimization.discounted_mdp import \
    state_successor_policy_evaluation_op, \
    state_distribution


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
