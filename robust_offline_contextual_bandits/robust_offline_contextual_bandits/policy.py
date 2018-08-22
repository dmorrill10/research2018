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


def sorted_values_across_worlds(policy, sampled_rewards):
    v = np.mean(
        np.mean(sampled_rewards * np.expand_dims(policy, axis=-1), axis=1),
        axis=0)
    v.sort()
    return v


def new_ff_policy_model(num_actions,
                        num_features,
                        num_units1=0,
                        num_units2=0,
                        num_hidden=0,
                        phi_f=None,
                        hidden_activation=tf.nn.relu,
                        output_activation=None):
    layers = []
    if phi_f is not None:
        layers.append(
            tf.keras.layers.Lambda(phi_f, output_shape=(None, num_features)))
    if num_hidden > 0:
        layers += [
            tf.keras.layers.Dense(
                num_units1 if i % 2 == 0 else num_units2,
                activation=hidden_activation,
                use_bias=True,
                input_shape=(None, num_features)) for i in range(num_hidden)
        ] + [
            tf.keras.layers.Dense(
                num_actions, use_bias=True, activation=output_activation)
        ]
    else:
        layers += [
            tf.keras.layers.Dense(
                num_actions,
                use_bias=False,
                input_shape=(None, num_features),
                kernel_initializer=tf.zeros_initializer(),
                bias_initializer=tf.zeros_initializer(),
                activation=output_activation)
        ]
    return tf.keras.Sequential(layers)
