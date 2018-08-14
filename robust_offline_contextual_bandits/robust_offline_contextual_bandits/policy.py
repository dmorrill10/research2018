import numpy as np
import tensorflow as tf


def greedy_policy(rewards):
    policy = np.zeros([rewards.shape[0]]).astype('int32')
    value = rewards[:, 0]
    for a in range(1, rewards.shape[1]):
        y = rewards[:, a]
        a_better = y > value
        policy[a_better] = a
        value[a_better] = y[a_better]
    return tf.one_hot(policy, rewards.shape[1])
