import tensorflow as tf
import numpy as np
import GPy as gp_lib
import os
from glob import glob


class GpMap(object):
    def __init__(self, gp):
        self.gp = gp

    def predict(self, phi):
        return self(phi)

    def __call__(self, phi):
        return self.gp.predict_noiseless(phi)[0].astype('float32')


class MultivariateNormalSampler(object):
    def __init__(self, mu, cov):
        self.mu = tf.expand_dims(mu, axis=0)
        self.scale_triu = tf.transpose(tf.cholesky(cov))

    def sample(self, n=1):
        return (self.mu + tf.random_normal(shape=[n, self.mu.shape[1].value])
                @ self.scale_triu)

    def num_outputs(self):
        return self.mu.shape[1].value


class GpAtInputs(object):
    @classmethod
    def load(cls, name):
        return cls(*np.load('{}.npy'.format(name)))

    @classmethod
    def load_all(cls, pattern):
        return {
            file: cls.load(os.path.splitext(file)[0])
            for file in glob(pattern)
        }

    def __init__(self, mean, var, full_cov=True, maxtries=10):
        self.mean = mean
        self.var = var
        self.maxtries = maxtries
        self.full_cov = full_cov

        output_dim = self.mean.shape[1]
        self.dists = []
        for d in range(output_dim):
            m = self.mean[:, d]
            adjusted_mean = m.flatten().astype('float32')

            if full_cov and self.var.ndim == 3:
                v = self.var[:, :, d]
            elif (not full_cov) and self.var.ndim == 2:
                v = self.var[:, d]
            else:
                v = self.var

            var_diag = np.diag(self.var)
            if np.any(var_diag <= 0.0):
                print(
                    "WARNING: not pd: non-positive diagonal elements: {}. Shifting diagonal.".
                    format(min(var_diag)))
                self.var = self.var - np.diag(np.minimum(var_diag, 0.0))
            jitter = var_diag.mean() * 1e-6
            num_tries = 1
            while num_tries <= maxtries and np.isfinite(jitter):
                try:
                    if full_cov:
                        self.dists.append(
                            MultivariateNormalSampler(
                                adjusted_mean, (v + np.eye(v.shape[0]) * jitter
                                                ).astype('float32')))
                    else:
                        self.dists.append(
                            MultivariateNormalSampler(
                                adjusted_mean, (v + jitter).astype('float32')))
                    return
                except:
                    jitter *= 10
                finally:
                    num_tries += 1
            raise RuntimeError("not positive definite, even with jitter.")

    def save(self, name):
        np.save('{}.npy'.format(name),
                [self.mean, self.var, self.full_cov, self.maxtries])
        return self

    def __call__(self, size=1):
        output_dim = self.mean.shape[1]
        if output_dim == 1:
            return tf.transpose(self.dists[0].sample(size))
        else:
            return tf.stack([
                tf.transpose(self.dists[d].sample(size))
                for d in range(output_dim)
            ])


class Gp(object):
    @classmethod
    def gp_regression(cls,
                      phi,
                      y,
                      inducing_fraction=1.0,
                      use_random_inducing=True,
                      kernel=None,
                      heteroscedastic=False):
        num_examples, num_features = phi.shape

        if kernel is None:
            kernel = (gp_lib.kern.Matern32(num_features) +
                      gp_lib.kern.White(num_features) +
                      gp_lib.kern.Bias(num_features))

        num_inducing = int(np.ceil(num_examples * inducing_fraction))

        kwargs = {}

        if not heteroscedastic:
            num_outputs = y.shape[1]
            bias = y.mean()
            mean_function = gp_lib.mappings.Constant(num_features, num_outputs,
                                                     bias)
            kwargs['mean_function'] = mean_function

        if num_inducing == num_examples:
            if heteroscedastic:
                new_gp = gp_lib.models.GPHeteroscedasticRegression
            else:
                new_gp = gp_lib.models.GPRegression
        else:
            if use_random_inducing:
                kwargs['Z'] = phi[np.random.choice(
                    num_examples, num_inducing, replace=False)]
            else:
                kwargs['num_inducing'] = num_inducing
            if heteroscedastic:
                new_gp = gp_lib.models.SparseGPHeteroscedasticRegression
            else:
                new_gp = gp_lib.models.SparseGPRegression
        return cls(new_gp(phi, y, kernel, **kwargs))

    def __init__(self, gp):
        self.gp = gp

    def sample(self, num_samples=1):
        return [self] * num_samples

    def predict(self, phi):
        return self(phi)

    def __call__(self, phi, size=1):
        return self.gp.posterior_samples_f(phi, size=size).astype('float32')

    def maximum_a_posteriori_estimate(self):
        return GpMap(self.gp)

    def at_inputs(self, phi):
        m, v = self.gp._raw_predict(phi, full_cov=True)
        if self.gp.normalizer is not None:
            m, v = (self.gp.normalizer.inverse_mean(m),
                    self.gp.normalizer.inverse_variance(v))
        return GpAtInputs(m, v)

    def train(self):
        self.gp.optimize()
        return self


def sample_reward_tensors_from_gp_function(num_worlds, phi, gp_models):
    gps_at_inputs = [model.at_inputs(phi) for model in gp_models]

    def f(n=num_worlds):
        return tf.transpose(
            tf.stack([model(size=n) for model in gps_at_inputs]), [1, 0, 2])

    return f


def map_predictions(gp_models, x):
    return tf.concat(
        [model.maximum_a_posteriori_estimate()(x) for model in gp_models],
        axis=1)
