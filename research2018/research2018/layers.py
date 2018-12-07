import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions


class ResMixin(object):
    def __init__(self, *args, residual_weight=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.residual_weight = residual_weight

    def call(self, inputs):
        return super().call(inputs) + self.residual_weight * inputs


def mvn_posterior_fn(dtype,
                     shape,
                     name,
                     trainable,
                     add_variable_fn,
                     share_cov=True):
    if len(shape) > 1:
        batch_size = shape[1]
        dimensionality = shape[0]

        loc = add_variable_fn(
            'loc',
            shape=[batch_size, dimensionality],
            dtype=dtype,
            trainable=trainable,
            initializer=tf.keras.initializers.zeros())

        if share_cov:
            return tfd.MultivariateNormalTriL(
                loc=loc,
                scale_tril=add_variable_fn(
                    'scale_tril',
                    shape=[dimensionality, dimensionality],
                    dtype=dtype,
                    trainable=trainable,
                    initializer=lambda shape, *args, **kwargs: (
                        1e-10 * tf.eye(*shape[1:], batch_shape=[1])
                    )
                )
            )
        else:
            return tfd.MultivariateNormalTriL(
                loc=loc,
                scale_tril=add_variable_fn(
                    'scale_tril',
                    shape=[batch_size, dimensionality, dimensionality],
                    dtype=dtype,
                    trainable=trainable,
                    initializer=lambda shape, *args, **kwargs: (
                        1e-10 * tf.eye(*shape[1:], batch_shape=shape[0:1])
                    )
                )
            )
    else:
        return tfd.Normal(
            loc=add_variable_fn(
                'loc',
                shape=[shape[0]],
                dtype=dtype,
                trainable=trainable,
                initializer=tf.keras.initializers.zeros()),
            scale=add_variable_fn(
                'scale',
                shape=[shape[0]],
                dtype=dtype,
                trainable=trainable,
                initializer=(
                    lambda shape, dtype, partition_info: tf.fill(shape, 1e-10)
                )))


mvn_posterior_fn_shared_cov = mvn_posterior_fn


def mvn_posterior_fn_independent_cov(*args, **kwargs):
    return mvn_posterior_fn(*args, **kwargs, share_cov=False)


def mvn_sample(d):
    return tf.transpose(d.sample())


def mvn_prior(dtype, shape, name, trainable, add_variable_fn, scale=1.0):
    # TODO: Why is trainable True here?
    if len(shape) > 1:
        return tfd.MultivariateNormalDiag(
            tf.zeros(reversed(shape), dtype=dtype),
            scale_identity_multiplier=scale,
            name=name)
    else:
        return tfd.Normal(
            tf.zeros(shape[0], dtype=dtype), scale=scale, name=name)


class DenseMvn(tfp.layers.DenseReparameterization):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            kernel_prior_fn=mvn_prior,
            kernel_posterior_fn=mvn_posterior_fn_independent_cov,
            kernel_posterior_tensor_fn=mvn_sample,
            bias_prior_fn=mvn_prior,
            bias_posterior_fn=mvn_posterior_fn_independent_cov,
            bias_posterior_tensor_fn=mvn_sample,
            **kwargs)


class DenseMvnSharedCov(tfp.layers.DenseReparameterization):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            kernel_prior_fn=mvn_prior,
            kernel_posterior_fn=mvn_posterior_fn_shared_cov,
            kernel_posterior_tensor_fn=mvn_sample,
            bias_prior_fn=mvn_prior,
            bias_posterior_fn=mvn_posterior_fn_shared_cov,
            bias_posterior_tensor_fn=mvn_sample,
            **kwargs)


class ResDenseMvnSharedCov(ResMixin, DenseMvnSharedCov):
    pass


class ResDenseFlipout(ResMixin, tfp.layers.DenseFlipout):
    pass


class ResConvolution2DFlipout(ResMixin, tfp.layers.Convolution2DFlipout):
    pass
