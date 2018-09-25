import numpy as np
import tensorflow as tf
from robust_offline_contextual_bandits.tf_np import logical_or


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


def new_low_rank_ff_policy_model(num_actions,
                                 num_features,
                                 num_units,
                                 initial_expansion=True,
                                 rank=1,
                                 num_hidden=1,
                                 hidden_activation=tf.nn.relu,
                                 output_activation=None):
    if rank >= num_units:
        return new_ff_policy_model(
            num_actions,
            num_features,
            num_units1=num_units,
            num_units2=num_units,
            num_hidden=num_hidden,
            hidden_activation=hidden_activation,
            output_activation=output_activation)
    else:
        layers = []
        if initial_expansion:
            layers.append(
                tf.keras.layers.Dense(
                    num_units,
                    activation=hidden_activation,
                    use_bias=True,
                    input_shape=(None, num_features)))
            num_hidden -= 1
        for i in range(num_hidden):
            layers.append(
                tf.keras.layers.Dense(
                    rank, use_bias=True, input_shape=(None, num_features)))
            layers.append(
                tf.keras.layers.Dense(
                    num_units, activation=hidden_activation, use_bias=True))

        layers.append(
            tf.keras.layers.Dense(
                num_actions, use_bias=True, activation=output_activation))
        return tf.keras.Sequential(layers)


def uniform_random_or_policy(rows_to_play_random, policy):
    policy = policy.copy()
    policy[rows_to_play_random] = 1.0 / policy.shape[1]
    return policy


def max_robust_policy(inputs_known_on_each_action, rewards_on_known_inputs):
    all_rewards = np.full([
        inputs_known_on_each_action[0].shape[0],
        len(inputs_known_on_each_action)
    ], -np.inf)
    for a, known_inputs in enumerate(inputs_known_on_each_action):
        if known_inputs.sum() > 0:
            all_rewards[known_inputs, a] = np.squeeze(
                rewards_on_known_inputs[a])

    policy = greedy_policy(all_rewards)

    rows_to_play_random = np.logical_not(
        logical_or(*inputs_known_on_each_action))
    if rows_to_play_random.sum() > 0:
        policy[rows_to_play_random] = 1.0 / policy.shape[1].value
    return policy
