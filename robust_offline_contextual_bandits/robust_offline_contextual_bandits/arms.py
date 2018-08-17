import tensorflow as tf
import numpy as np
import GPy as gp_lib
from tf_contextual_prediction_with_expert_advice import utility

from robust_offline_contextual_bandits.plotting import tableu20_color_table
from robust_offline_contextual_bandits.gp import Gp


class Arms(object):
    def __init__(self, reward_functions, colors=None):
        self.reward_functions = reward_functions
        if colors is None:
            color_table = tableu20_color_table()
            colors = [next(color_table) for _ in reward_functions]
        self.colors = colors
        assert len(self.reward_functions) == len(self.colors)

    def components_for_training(self, x, stddev=0.0):
        return [f.for_training(x, stddev) for f in self.reward_functions]

    def __len__(self):
        return len(self.reward_functions)

    def with_function_outside_plateaus(self, f):
        return self.__class__(
            [
                r.with_function_outside_plateaus(f)
                for r in self.reward_functions
            ],
            colors=self.colors)


class ArmsWithContexts(object):
    def __init__(self, arms, x, stddev=0.0):
        self.arms = arms
        self.x = x
        self.stddev = stddev
        self.components_for_training = arms.components_for_training(x, stddev)

    def value_slope_and_bias(self, policy):
        test_rewards = [
            self.with_function_outside_plateaus(
                lambda x: np.full(x.shape, float(i))).combined_raw_y()
            for i in range(2)
        ]
        bias = tf.reduce_mean(utility(policy, test_rewards[0]))
        slope = tf.reduce_mean(utility(policy, test_rewards[1])) - bias
        return slope, bias

    def new_gp_models(self,
                      gp_inducing_input_fraction=1.0,
                      use_random_inducing=True):
        gp_models = []
        for a, components in self.training_components_from_each_arm():
            num_examples = components.noisy_data.good.num_examples()
            num_features = components.noisy_data.good.num_features()

            num_inducing = int(
                np.ceil(num_examples * gp_inducing_input_fraction))

            gp_models.append(
                Gp.gp_regression(
                    components.noisy_data.good.phi.numpy(),
                    components.noisy_data.good.y.numpy(),
                    gp_lib.kern.Matern32(num_features) +
                    gp_lib.kern.White(num_features),
                    num_inducing,
                    use_random_inducing=use_random_inducing))
        return gp_models

    @property
    def colors(self):
        return self.arms.colors

    @property
    def reward_functions(self):
        return self.arms.reward_functions

    def __getitem__(self, action):
        return self.components_for_training[action]

    def combined_raw_y(self, action=None):
        if action is None:
            return tf.concat(
                [self.combined_raw_y(a) for a in range(self.num_arms())],
                axis=1)
        else:
            si = self[action].sort_indices
            return self[action].combined_raw_data.y.numpy()[si, :]

    def num_contexts(self):
        return len(self.x)

    def num_arms(self):
        return len(self.arms)

    def with_function_outside_plateaus(self, f):
        return self.__class__(
            self.arms.with_function_outside_plateaus(f), self.x, self.stddev)

    def training_components_from_each_arm(self):
        for a in range(self.num_arms()):
            yield a, self[a]
