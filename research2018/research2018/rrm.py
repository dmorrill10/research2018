import tensorflow as tf


def normalized_by_sum(v, axis=0):
    """Divides each element of `v` along `axis` by the sum of `v` along `axis`.

  Adapted from the [OpenSpiel project](https://github.com/deepmind/open_spiel)
  (this function was authored by me, Dustin Morrill).

  Assumes `v` is non-negative. Sets of `v` elements along `axis` that sum to
  zero are normalized to `1 / v.shape[axis]` (a uniform distribution).

  Args:
    v: Non-negative tensor of values.
    axis: An integer axis.

  Returns:
    The normalized `tf.Tensor`.
  """
    v = tf.convert_to_tensor(v)
    denominator = tf.reduce_sum(v, axis=axis, keepdims=True)
    denominator_is_zero = tf.cast(tf.equal(denominator, 0), v.dtype)

    # Every element of `denominator_is_zero` that is true corresponds to a
    # set of elements in `v` along `axis` that are all zero. By setting these
    # denominators to `v.shape[axis]` and adding 1 to each of the corresponding
    # elements in `v`, these elements are normalized to `1 / v.shape[axis]`
    # (a uniform distribution).
    denominator = denominator + v.shape[axis] * denominator_is_zero
    return (v + denominator_is_zero) / denominator


def rm_policy(regrets):
    num_actions = regrets.shape[1]
    qregrets = tf.maximum(regrets, 0.0)
    z = tf.tile(tf.reduce_sum(qregrets, axis=1, keepdims=True),
                [1, num_actions])
    uniform_strat = tf.fill(tf.shape(qregrets), 1.0 / num_actions)
    return tf.where(tf.greater(z, 0.0), qregrets / z, uniform_strat)


def utility(policy, action_utilities):
    return tf.reduce_sum(policy * action_utilities, axis=1, keepdims=True)


def rrm_utilities(model, contexts, action_utilities):
    return utility(rm_policy(model(contexts)), action_utilities)


def rrm_loss(model, contexts, action_utilities):
    regrets = model(contexts)
    num_actions = regrets.shape[1]
    policy = rm_policy(regrets)

    inst_regret = tf.stop_gradient(
        action_utilities - tf.tile(
            utility(policy, action_utilities),
            [1, num_actions]
        )
    )  # yapf:disable

    is_substantive_regret_diff = tf.stop_gradient(
        tf.logical_or(
            tf.greater(regrets, 0.0),
            tf.greater(inst_regret, 0.0)))  # yapf:disable

    regret_diffs = tf.where(is_substantive_regret_diff,
                            tf.square(regrets - inst_regret),
                            tf.zeros_like(inst_regret))

    return tf.reduce_mean(regret_diffs) / 2.0


def rrm_grad(model, contexts, action_utilities):
    with tf.GradientTape() as tape:
        loss_value = rrm_loss(model, contexts, action_utilities)
    return zip(tape.gradient(loss_value, model.variables), model.variables)


def indmax(t):
    idx = tf.expand_dims(tf.argmax(t, axis=1, output_type=tf.int32), axis=1)
    num_contexts = tf.shape(idx)[0]
    return tf.scatter_nd(tf.concat(
        [tf.expand_dims(tf.range(num_contexts, dtype=tf.int32), axis=1), idx],
        axis=1),
                         tf.ones([num_contexts]),
                         shape=tf.shape(t))


def br(action_utilities):
    return indmax(action_utilities)


def behavioral_to_sequence_form_strat(policy):
    return tf.reshape(policy, [tf.size(policy), 1])
