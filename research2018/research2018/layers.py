import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfk = tf.keras


class ResMixin(object):
    def __init__(self, *args, input_transformation=lambda x: x, **kwargs):
        super().__init__(*args, **kwargs)
        self._input_transformation = input_transformation

    def call(self, inputs):
        return tfk.layers.add(
            [super().call(inputs),
             self._input_transformation(inputs)])


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


def mvn_mean(d):
    return tf.transpose(d.mean())


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


def deterministic_dense(var_dense_layer, tensor_fn=lambda d: d.mean()):
    use_bias = var_dense_layer.bias_posterior is not None
    return tfk.layers.Dense(
        var_dense_layer.units,
        activation=var_dense_layer.activation,
        use_bias=use_bias,
        kernel_initializer=(
            lambda *_, **__: tensor_fn(var_dense_layer.kernel_posterior)),
        bias_initializer=(
            lambda *_, **__: tensor_fn(var_dense_layer.bias_posterior)),
        activity_regularizer=var_dense_layer.activity_regularizer)


def deterministic_conv2d(var_conv2d_layer, tensor_fn=lambda d: d.mean()):
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
            lambda *_, **__: tensor_fn(var_conv2d_layer.kernel_posterior)),
        bias_initializer=(
            lambda *_, **__: tensor_fn(var_conv2d_layer.bias_posterior)),
        activity_regularizer=var_conv2d_layer.activity_regularizer)


def clone_layer(layer):
    return layer.__class__.from_config(layer.get_config())


def deterministic_layer(layer, tensor_fn=lambda d: d.mean()):
    try:
        return deterministic_dense(layer, tensor_fn)
    except:
        try:
            return deterministic_conv2d(layer, tensor_fn)
        except:
            return clone_layer(layer)


def deterministic_model(model, *tensor_fns):
    return model.__class__([
        deterministic_layer(layer, tensor_fns[i])
        for i, layer in enumerate(model.layers)
    ])


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


class ResConvolution2DReparameterization(
        ResMixin, tfp.layers.Convolution2DReparameterization):
    pass


ResConv2DReparameterization = ResConvolution2DReparameterization


class ResConvolution2DFlipout(ResMixin, tfp.layers.Convolution2DFlipout):
    pass


ResConv2DFlipout = ResConvolution2DFlipout


class ResDense(ResMixin, tf.keras.layers.Dense):
    pass


class ResConvolution2D(ResMixin, tf.keras.layers.Convolution2D):
    pass


ResConv2D = ResConvolution2D

