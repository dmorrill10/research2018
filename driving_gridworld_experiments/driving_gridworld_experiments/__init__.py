import tensorflow as tf
from driving_gridworld.road import Road
from driving_gridworld.car import Car
from driving_gridworld.obstacles import Bump, Pedestrian
from driving_gridworld.rewards import TfUniformSituationalReward


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
