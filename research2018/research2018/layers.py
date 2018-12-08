import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfk = tf.keras


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
            initializer=tfk.initializers.zeros())

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
                initializer=tfk.initializers.zeros()),
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


def mean_dense(var_dense_layer):
    use_bias = var_dense_layer.bias_posterior is not None
    return tfk.layers.Dense(
        var_dense_layer.units,
        activation=var_dense_layer.activation,
        use_bias=use_bias,
        kernel_initializer=(
            lambda *_, **__: var_dense_layer.kernel_posterior.mean()),
        bias_initializer=(
            lambda *_, **__: var_dense_layer.bias_posterior.mean()),
        activity_regularizer=var_dense_layer.activity_regularizer)


def sample_dense(var_dense_layer):
    use_bias = var_dense_layer.bias_posterior is not None
    return tfk.layers.Dense(
        var_dense_layer.units,
        activation=var_dense_layer.activation,
        use_bias=use_bias,
        kernel_initializer=(
            lambda *_, **__: var_dense_layer.kernel_posterior.sample()),
        bias_initializer=(
            lambda *_, **__: var_dense_layer.bias_posterior.sample()),
        activity_regularizer=var_dense_layer.activity_regularizer)


def mean_conv2d(var_conv2d_layer):
    use_bias = var_conv2d_layer.bias_posterior is not None
    return tfk.layers.Conv2D(
        var_conv2d_layer.filters,
        var_conv2d_layer.kernel_size,
        strides=var_conv2d_layer.strides,
        padding=var_conv2d_layer.padding,
        data_format=var_conv2d_layer.data_format,
        dilation_rate=var_conv2d_layer.dilation_rate,
        activation=var_conv2d_layer.activation,
        use_bias=use_bias,
        kernel_initializer=(
            lambda *_, **__: var_conv2d_layer.kernel_posterior.mean()),
        bias_initializer=(
            lambda *_, **__: var_conv2d_layer.bias_posterior.mean()),
        activity_regularizer=var_conv2d_layer.activity_regularizer)


def sample_conv2d(var_conv2d_layer):
    use_bias = var_conv2d_layer.bias_posterior is not None
    return tfk.layers.Conv2D(
        var_conv2d_layer.filters,
        var_conv2d_layer.kernel_size,
        strides=var_conv2d_layer.strides,
        padding=var_conv2d_layer.padding,
        data_format=var_conv2d_layer.data_format,
        dilation_rate=var_conv2d_layer.dilation_rate,
        activation=var_conv2d_layer.activation,
        use_bias=use_bias,
        kernel_initializer=(
            lambda *_, **__: var_conv2d_layer.kernel_posterior.sample()),
        bias_initializer=(
            lambda *_, **__: var_conv2d_layer.bias_posterior.sample()),
        activity_regularizer=var_conv2d_layer.activity_regularizer)


def clone_layer(layer):
    return layer.__class__.from_config(layer.get_config())


def mean_model(model):
    def mean_layer(layer):
        try:
            return mean_dense(layer)
        except:
            try:
                return mean_conv2d(layer)
            except:
                return clone_layer(layer)

    return model.__class__([mean_layer(layer) for layer in model.layers])


def sample_model(model):
    def sample_layer(layer):
        try:
            return sample_dense(layer)
        except:
            try:
                return sample_conv2d(layer)
            except:
                return clone_layer(layer)

    return model.__class__([sample_layer(layer) for layer in model.layers])


class DenseMvn(tfp.layers.DenseReparameterization):
    def __init__(self, *args, **kwargs):
        kwargs['kernel_prior_fn'] = mvn_prior
        kwargs['kernel_posterior_fn'] = mvn_posterior_fn_independent_cov
        kwargs['kernel_posterior_tensor_fn'] = mvn_sample
        kwargs['bias_prior_fn'] = mvn_prior
        kwargs['bias_posterior_fn'] = mvn_posterior_fn_independent_cov
        kwargs['bias_posterior_tensor_fn'] = mvn_sample
        super().__init__(*args, **kwargs)


class DenseMvnSharedCov(tfp.layers.DenseReparameterization):
    def __init__(self, *args, **kwargs):
        kwargs['kernel_prior_fn'] = mvn_prior
        kwargs['kernel_posterior_fn'] = mvn_posterior_fn_shared_cov
        kwargs['kernel_posterior_tensor_fn'] = mvn_sample
        kwargs['bias_prior_fn'] = mvn_prior
        kwargs['bias_posterior_fn'] = mvn_posterior_fn_shared_cov
        kwargs['bias_posterior_tensor_fn'] = mvn_sample
        super().__init__(*args, **kwargs)


class ResDenseMvnSharedCov(ResMixin, DenseMvnSharedCov):
    pass


class ResDenseFlipout(ResMixin, tfp.layers.DenseFlipout):
    pass


class ResConvolution2DFlipout(ResMixin, tfp.layers.Convolution2DFlipout):
    pass
