import tensorflow as tf
import numpy as np


class ResMixin(object):
    def call(self, inputs):
        return super().call(inputs) + inputs


class FixedDense(tf.keras.layers.Layer):
    def __init__(self, kernel, bias, **kwargs):
        self.kernel = tf.convert_to_tensor(kernel)
        self.bias = tf.convert_to_tensor(bias)
        super(FixedDense, self).__init__(**kwargs)

    def call(self, inputs):
        return inputs @ self.kernel + self.bias

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.kernel.shape[-1].value
        return tf.TensorShape(shape)


class NoisyDense(tf.keras.layers.Layer):
    def __init__(
            self,
            output_dim,
            mu_initializer='zeros',
            sigma_initializer=(
                lambda shape, *args, **kwargs: tf.eye(shape[0].value, shape[1].value)
            ),
            activation=tf.identity,
            **kwargs):
        self.output_dim = output_dim
        self.mu_initializer = mu_initializer
        self.sigma_initializer = sigma_initializer
        self.activation = activation
        super(NoisyDense, self).__init__(**kwargs)

    def build(self, input_shape):
        self.mu_kernel = self.add_weight(
            name='mu_kernel',
            shape=tf.TensorShape((input_shape[1], self.output_dim)),
            initializer=self.mu_initializer,
            trainable=True)
        self.mu_bias = self.add_weight(
            name='mu_bias',
            shape=tf.TensorShape((1, self.output_dim)),
            initializer=self.mu_initializer,
            trainable=True)

        def sigma_constraint(k):
            k_diag = tf.diag_part(k)
            k_diag = tf.where(
                tf.equal(k_diag, 0.0), tf.fill(k_diag.shape, 1e-5), k_diag)
            return tf.linalg.set_diag(k, k_diag)

        self.sigma_kernel = self.add_weight(
            name='sigma_kernel',
            shape=tf.TensorShape((input_shape[1], input_shape[1])),
            initializer=self.sigma_initializer,
            constraint=sigma_constraint,
            trainable=True)
        self.sigma_bias = self.add_weight(
            name='sigma_bias',
            shape=tf.TensorShape((1, 1)),
            initializer=self.sigma_initializer,
            constraint=sigma_constraint,
            trainable=True)
        return super(NoisyDense, self).build(input_shape)

    def L_kernel(self):
        return tf.linalg.LinearOperatorLowerTriangular(self.sigma_kernel)

    def L_bias(self):
        return tf.linalg.LinearOperatorLowerTriangular(self.sigma_bias)

    def kernel(self,
               standard_normal=lambda shape: tf.random_normal(shape=shape)):
        return self.mu_kernel + self.L_kernel().matmul(
            standard_normal(self.mu_kernel.shape))

    def bias(self,
             standard_normal=lambda shape: tf.random_normal(shape=shape)):
        return self.mu_bias + self.L_bias().matmul(
            standard_normal(self.mu_bias.shape))

    def call(self, inputs):
        return self.activation(inputs @ self.kernel() + self.bias())

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.output_dim
        return tf.TensorShape(shape)

    def sample(self):
        return FixedDense(
            self.kernel(
                lambda shape: np.random.normal(size=shape).astype('float32')),
            self.bias(
                lambda shape: np.random.normal(size=shape).astype('float32')))

    def entropy_cov_part(self):
        return sum([
            L.log_abs_determinant()
            for L in [self.L_kernel(), self.L_bias()]
        ]) / 2.0


class ResNoisyDense(ResMixin, NoisyDense):
    pass
