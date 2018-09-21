import tensorflow as tf


def sample_action(policy):
    dist = tf.distributions.Categorical(probs=policy)
    return dist.sample(1)
