import tensorflow as tf
import numpy as np


class ResMixin(object):
    def call(self, inputs):
        return super().call(inputs) + inputs


class FixedDense(tf.keras.layers.Layer):
    def __init__(self, kernel, bias, activation=lambda z: z, **kwargs):
        self.kernel = tf.convert_to_tensor(kernel)
        self.bias = tf.convert_to_tensor(bias)
        self.activation = activation
        super(FixedDense, self).__init__(**kwargs)

    def call(self, inputs):
        return self.activation(inputs @ self.kernel + self.bias)

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.kernel.shape[-1].value
        return tf.TensorShape(shape)


class LnoisyDense(tf.keras.layers.Layer):
    def __init__(
            self,
            output_dim,
            mu_initializer='zeros',
            sigma_initializer=tf.eye,
            activation=lambda z: z,
            sigma_trainable=True,
            share_cov=True,
            **kwargs):
        self.output_dim = output_dim
        self.mu_initializer = mu_initializer
        self.sigma_initializer = sigma_initializer
        self.activation = activation
        self.sigma_trainable = sigma_trainable
        self.share_cov = share_cov
        super(LnoisyDense, self).__init__(**kwargs)

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()

        self.mu_kernel = self.add_weight(
            name='mu_kernel',
            shape=(input_shape[1], self.output_dim),
            initializer=self.mu_initializer,
            trainable=True)
        self.mu_bias = self.add_weight(
            name='mu_bias',
            shape=(1, self.output_dim),
            initializer='zeros',
            trainable=True)

        if self.share_cov:
            self.sigma_kernel = self.add_weight(
                name='sigma_kernel',
                shape=(input_shape[1], input_shape[1]),
                initializer=self.sigma_initializer,
                trainable=self.sigma_trainable)

            self.sigma_bias = self.add_weight(
                name='sigma_bias',
                shape=(1, 1),
                initializer=self.sigma_initializer,
                trainable=self.sigma_trainable)
        else:
            self.sigma_kernel = self.add_weight(
                name='sigma_kernel',
                shape=(self.output_dim, input_shape[1], input_shape[1]),
                initializer=self.sigma_initializer,
                trainable=self.sigma_trainable)

            self.sigma_bias = self.add_weight(
                name='sigma_bias',
                shape=(self.output_dim, 1, 1),
                initializer=self.sigma_initializer,
                trainable=self.sigma_trainable)
        return super(LnoisyDense, self).build(input_shape)

    def L_kernel(self):
        return tf.linalg.LinearOperatorLowerTriangular(self.sigma_kernel)

    def L_bias(self):
        return tf.linalg.LinearOperatorLowerTriangular(self.sigma_bias)

    def kernel(self,
               standard_normal=lambda shape: tf.random_normal(shape=shape)):
        if self.share_cov:
            randomness = self.L_kernel().matmul(self.mu_kernel)
        else:
            randomness = tf.transpose(self.L_kernel().matvec(
                standard_normal(list(reversed(self.mu_kernel.shape)))))
        return self.mu_kernel + randomness

    def bias(self,
             standard_normal=lambda shape: tf.random_normal(shape=shape)):
        if self.share_cov:
            randomness = self.L_bias().matmul(self.mu_bias)
        else:
            randomness = tf.transpose(self.L_bias().matvec(
                standard_normal(list(reversed(self.mu_bias.shape)))))
        return self.mu_bias + randomness

    def call(self, inputs, sample=True):
        if sample:
            k = self.kernel()
            b = self.bias()
        else:
            k = self.mu_kernel
            b = self.mu_bias
        return self.activation(inputs @ k + b)

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.output_dim
        return tf.TensorShape(shape)

    def sample(self):
        return FixedDense(
            self.kernel(
                lambda shape: np.random.normal(size=shape).astype('float32')),
            self.bias(
                lambda shape: np.random.normal(size=shape).astype('float32')),
            activation=self.activation)

    def entropy_cov_part(self):
        aggregate = tf.stack if self.share_cov else tf.concat
        return tf.reduce_sum(
            aggregate(
                [
                    L.log_abs_determinant()
                    for L in [self.L_kernel(), self.L_bias()]
                ],
                axis=0)) / 2.0


class ResLnoisyDense(ResMixin, LnoisyDense):
    pass
