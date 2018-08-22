import tensorflow as tf
import numpy as np
import GPy as gp_lib


class GpMap(object):
    def __init__(self, gp, bias=0.0):
        self.gp = gp
        self.bias = bias

    def predict(self, phi):
        return self(phi)

    def __call__(self, phi):
        return self.gp.predict_noiseless(phi)[0].astype('float32') + self.bias


class MultivariateNormalSampler(object):
    def __init__(self, mu, cov):
        self.mu = tf.expand_dims(mu, axis=0)
        self.scale_triu = tf.transpose(tf.cholesky(cov))

    def sample(self, n=1):
        return (self.mu + tf.random_normal(shape=[n, self.mu.shape[1].value])
                @ self.scale_triu)


class GpAtInputs(object):
    def __init__(self, mean, var, bias=0.0, full_cov=True, maxtries=10):
        self.mean = mean
        self.var = var
        self.bias = bias
        self.full_cov = full_cov

        output_dim = self.mean.shape[1]
        self.dists = []
        for d in range(output_dim):
            m = self.mean[:, d]
            adjusted_mean = m.flatten().astype('float32') + self.bias

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
                # self.var = np.maximum(self.var, 0.0)
            jitter = var_diag.mean() * 1e-6
            num_tries = 1
            while num_tries <= maxtries and np.isfinite(jitter):
                try:
                    if self.full_cov:
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
                      kernel,
                      num_inducing,
                      use_random_inducing=True):
        num_examples, num_features = phi.shape
        num_outputs = phi.shape[1]
        bias = y.mean()
        mean_function = gp_lib.mappings.Constant(num_features, num_outputs,
                                                 bias)

        if num_inducing == num_examples:
            gp = gp_lib.models.GPRegression(
                phi, y, kernel, mean_function=mean_function)
        else:
            gp = (
                gp_lib.models.SparseGPRegression(
                    phi,
                    y,
                    kernel,
                    Z=phi[np.random.choice(
                        num_examples, num_inducing, replace=False)],
                    mean_function=mean_function
                ) if use_random_inducing
                else gp_lib.models.SparseGPRegression(
                    phi,
                    y,
                    kernel,
                    num_inducing=num_inducing,
                    mean_function=mean_function
                )
            )  # yapf:disable
        return cls(gp)

    def __init__(self, gp, bias=0.0):
        self.gp = gp
        self.bias = bias

    def sample(self, num_samples=1):
        return [self] * num_samples

    def predict(self, phi):
        return self(phi)

    def __call__(self, phi, size=1):
        return (self.gp.posterior_samples_f(phi, size=size).astype('float32') +
                self.bias)

    def maximum_a_posteriori_estimate(self):
        return GpMap(self.gp, bias=self.bias)

    def at_inputs(self, phi):
        m, v = self.gp._raw_predict(phi, full_cov=True)
        if self.gp.normalizer is not None:
            m, v = (self.gp.normalizer.inverse_mean(m),
                    self.gp.normalizer.inverse_variance(v))
        return GpAtInputs(m, v, bias=self.bias)


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


def new_gp_models(training_data_generator,
                  gp_inducing_input_fraction=1.0,
                  use_random_inducing=True):
    gp_models = []
    for a, (x_train, y_train) in enumerate(training_data_generator):
        num_examples = len(x_train)
        num_features = x_train.shape[1]

        num_inducing = int(np.ceil(num_examples * gp_inducing_input_fraction))

        gp_models.append(
            Gp.gp_regression(
                x_train,
                y_train,
                gp_lib.kern.Matern32(num_features) +
                gp_lib.kern.White(num_features),
                num_inducing,
                use_random_inducing=use_random_inducing))
    return gp_models
