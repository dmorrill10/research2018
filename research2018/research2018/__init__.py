import tensorflow as tf
import numpy as np


def sample_action_tf(policy, num_samples=1):
    return tf.distributions.Categorical(probs=policy).sample(num_samples)


def sample_action_np(policy_at_cur_state, num_samples=1):
    size = [num_samples] if num_samples > 1 else []
    return np.random.choice(
        a=range(len(policy_at_cur_state)), size=size, p=policy_at_cur_state)
