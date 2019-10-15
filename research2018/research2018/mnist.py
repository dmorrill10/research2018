import tensorflow as tf
# import tensorflow_probability as tfp
import numpy as np

# from research2018 import layers
import tf_contextual_prediction_with_expert_advice as cpea
from research2018 import data


class DataNormalizer(object):
    def __init__(self, x):
        self._abs_min = np.abs(x).min(axis=0, keepdims=True).astype('float32')
        self._abs_max = np.abs(x).max(axis=0, keepdims=True).astype('float32')
        self._mean = x.mean(axis=0, keepdims=True).astype('float32')

    def __call__(self, x):
        raise NotImplementedError("Please override.")


class SubtractMeanDivideByScaleDataNormalizer(DataNormalizer):
    @tf.function
    def __call__(self, x):
        return data.subtract_mean_divide_by_scale(tf.cast(x, tf.float32),
                                                  self._abs_min, self._abs_max,
                                                  self._mean)


def perturbed_img(img, perturbation=None, max_change=100):
    if perturbation is None:
        perturbation = np.random.uniform(low=0,
                                         high=max_change,
                                         size=[1] +
                                         list(img.shape)).astype('float32')
    return np.minimum(np.maximum(img + perturbation, 0),
                      255).round().astype('uint8')


class ImageClassRewardDataset(data.Dataset):
    @classmethod
    def new(cls,
            x,
            y,
            num_classes,
            reward_fn=lambda y, yh: (y == yh).astype('float32'),
            constant_reward=None):
        if constant_reward is None:
            rewards = np.array(
                [reward_fn(y, y_hat) for y_hat in range(num_classes)],
                dtype='float32').T
        else:
            rewards = np.array(
                [reward_fn(y, y_hat) for y_hat in range(num_classes)] +
                [np.full(y.shape, constant_reward)],
                dtype='float32').T
        return cls(x, y, rewards)

    def perturbed(self, perturbation=None):
        return self.__class__(perturbed_img(self.x, perturbation=perturbation),
                              self.y, self.reward)

    @tf.function
    def avg_prediction_mean_std(self, sample_models, batch_size=None, seed=42):
        avg_mean = tf.zeros([self.reward.shape[1]])
        avg_std = tf.zeros([self.reward.shape[1]])

        if batch_size is None:
            batch_size = len(self) // 10
        data = self.clone()
        data = data.batch(batch_size).repeat(1)

        i = 0
        for x, _ in data:
            i += 1
            samples = [model(x) for model in sample_models]
            m = tf.reduce_mean(samples, axis=0)
            s = tf.math.reduce_std(samples, axis=0)
            avg_mean += (tf.reduce_mean(m, axis=0) - avg_mean) / float(i)
            avg_std += (tf.reduce_mean(s, axis=0) - avg_std) / float(i)
        return avg_mean, avg_std

    @tf.function
    def loss(self, model, batch_size=None):
        if batch_size is None:
            batch_size = len(self) // 10
        data = self.clone()
        data = data.batch(batch_size).repeat(1)
        batch_loss = 0.0
        i = 0
        for x, rewards in data:
            i += 1
            next_reward = tf.reduce_mean(cpea.utility(model(x), rewards))
            batch_loss += -(next_reward + batch_loss) / float(i)
        return batch_loss


def new_training_dataset_and_normalizer(constant_reward=None):
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()

    # Add color channel
    x_train = np.expand_dims(x_train, -1)

    dataset = ImageClassRewardDataset.new(x=x_train,
                                          y=y_train,
                                          num_classes=10,
                                          constant_reward=constant_reward)
    return dataset, SubtractMeanDivideByScaleDataNormalizer(x_train)


# class KernelPrior(object):
#     def __init__(self, stddev=1):
#         self.stddev = stddev
#
#     def output(self, dtype, shape, name, trainable, add_variable_fn):
#         scale = np.full(shape, self.stddev, dtype=dtype.as_numpy_dtype)
#         dist = tfp.distributions.Normal(loc=tf.zeros(shape, dtype),
#                                         scale=scale)
#         batch_ndims = tf.size(dist.batch_shape_tensor())
#         return tfp.distributions.Independent(
#             dist, reinterpreted_batch_ndims=batch_ndims)
#
#     def conv(self, dtype, shape, name, trainable, add_variable_fn):
#         dist = tfp.distributions.Normal(loc=tf.zeros(shape, dtype),
#                                         scale=dtype.as_numpy_dtype(
#                                             self.stddev))
#         batch_ndims = tf.size(dist.batch_shape_tensor())
#         return tfp.distributions.Independent(
#             dist, reinterpreted_batch_ndims=batch_ndims)
#
#
# def weighted_divergence_fn(log_weight):
#     def divergence_fn(pos, pri):
#         return (tf.exp(float(log_weight)) *
#                 tf.reduce_mean(pos.kl_divergence(pri)))
#
#     return divergence_fn
#
#
# def new_deep_linear_bnn(data_normalizer,
#                         num_actions,
#                         filters=8,
#                         log_divergence_weight=-3,
#                         prior_stddev=1,
#                         residual_weight=0.1,
#                         log_log_divergence_weight=0.0):
#     '''
#     Creates a new Bayesian neural network on images.
#
#     Arguments:
#     - data_normalizer: A function that takes MNIST images preprocesses them.
#     - filters: The number of convolutional filters in the two hidden
#         convolutional layers. Defaults to 8 just because I saw another script
#         use this many.
#     - log_divergence_weight: The weight of the divergence penalty on each
#         layer. Defaults to -3 since that worked best for me with MNIST.
#     - prior_stddev: The standard deviation of the prior weight distributions.
#         Defaults to 1 since that should probably be a good place to start.
#         Might need to turn this up a lot though to get the layers to be more
#         random, so you could set this as large as 20.
#     - residual_weight: The weight on the residual term in the residual
#         layers. Defaults to 0.1 since that worked best for me. You can set it
#         to zero to make the layers non-residual.
#     '''
#     return tf.keras.Sequential([
#         tf.keras.layers.Lambda(data_normalizer),
#         layers.ResConvolution2D(
#             filters=filters,
#             kernel_size=3,
#             padding='SAME',
#             activation=tf.nn.relu,
#             residual_weight=residual_weight,
#             kernel_prior_fn=KernelPrior(prior_stddev).conv,
#             kernel_divergence_fn=weighted_divergence_fn(log_divergence_weight),
#             bias_posterior_fn=tfp.layers.default_mean_field_normal_fn(
#                 is_singular=False, loc_initializer=tf.zeros_initializer()),
#             bias_divergence_fn=weighted_divergence_fn(log_divergence_weight)),
#         layers.ResConvolution2D(
#             filters=filters,
#             kernel_size=5,
#             padding='SAME',
#             activation=tf.nn.relu,
#             residual_weight=residual_weight,
#             kernel_prior_fn=KernelPrior(prior_stddev).conv,
#             kernel_divergence_fn=weighted_divergence_fn(
#                 log_log_divergence_weight),
#             bias_posterior_fn=tfp.layers.default_mean_field_normal_fn(
#                 is_singular=False, loc_initializer=tf.zeros_initializer()),
#             bias_divergence_fn=weighted_divergence_fn(log_divergence_weight)),
#         tf.keras.layers.AveragePooling2D(pool_size=[2, 2],
#                                          strides=[2, 2],
#                                          padding='SAME'),
#         tf.keras.layers.Flatten(),
#         tfp.layers.DenseFlipout(
#             num_actions,
#             kernel_prior_fn=KernelPrior(prior_stddev).output,
#             kernel_divergence_fn=weighted_divergence_fn(log_divergence_weight),
#             bias_posterior_fn=tfp.layers.default_mean_field_normal_fn(
#                 is_singular=False, loc_initializer=tf.zeros_initializer()),
#             bias_divergence_fn=weighted_divergence_fn(log_divergence_weight))
#     ])
