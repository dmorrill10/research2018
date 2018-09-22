import tensorflow as tf
import numpy as np
from robust_offline_contextual_bandits.data import load_or_save, load_list
from robust_offline_contextual_bandits import cache
from robust_offline_contextual_bandits.tf_np import reset_random
from robust_offline_contextual_bandits.plotting import tableu20_color_table
from robust_offline_contextual_bandits.policy import \
    max_robust_policy, \
    greedy_policy
from robust_offline_contextual_bandits.gp import new_gp_models
from robust_offline_contextual_bandits.plateau_function import PlateauFunction


@load_list
def load_all_plateau_functions(reality_idx):
    return list(
        zip(*sorted(
            PlateauFunction.load_all('plateau_function.{}.*'.format(
                reality_idx)).items(),
            key=lambda file: int(file.split('.')[-2]))))[1]


class RealityExperiment(object):
    def __init__(self,
                 id,
                 num_actions,
                 x_train,
                 x_test,
                 x_known_on_each_action=None):
        if x_known_on_each_action is None:
            x_known_on_each_action = [
                np.full([len(x_train)], True) for _ in range(num_actions)
            ]
        self.id = id
        self.num_actions = num_actions
        self.x_train = x_train
        self.x_test = x_test
        self.x_known_on_each_action = [
            np.concatenate(
                [x_known, np.full([len(self.x_test)], False)], axis=0)
            for x_known in x_known_on_each_action
        ]
        for xk in self.x_known_on_each_action:
            assert len(xk) == len(self.x)

    @cache
    def x(self):
        return np.concatenate([self.x_train, self.x_test], axis=0)

    @cache
    def num_training_examples(self):
        return self.x_known_on_each_action.sum(axis=0)

    def reset_random(self):
        reset_random(self.id)

    def reward_functions(self):
        raise NotImplementedError('Override')

    def action_colors(self):
        color_table = tableu20_color_table()
        return [next(color_table) for _ in range(self.num_actions)]

    @cache
    def max_robust_policy(self):
        return load_or_save(
            lambda: np.load('max_robust_policy.{}.npy'.format(self.id)),
            lambda policy: np.save('max_robust_policy.{}.npy'.format(self.id), policy)
        )(max_robust_policy)(self.x_known_on_each_action,
                             [r(self.x) for r in self.reward_functions()])

    @cache
    def map_policy(self):
        def compute_map_policy():
            return greedy_policy(
                np.concatenate(
                    [model.mean for model in self.gp_at_inputs],
                    axis=1)).numpy()

        return load_or_save(
            lambda: np.load('map_policy.{}.npy'.format(self.id)),
            lambda policy: np.save('map_policy.{}.npy'.format(self.id), policy)
        )(compute_map_policy)()

    def reward_sampler(self, x):
        rfds = self.reward_function_distributions(x)

        def sample_rewards(n):
            return tf.transpose(
                tf.stack([action_model(n) for action_model in rfds]),
                [1, 0, 2])

        return sample_rewards

    def reward_function_distributions(self, x):
        raise NotImplementedError('Override')


class GpRealityExperimentMixin(object):
    def __init__(self, train_gp_model, gp_inducing_input_fraction, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.train_gp_model = train_gp_model
        self.gp_inducing_input_fraction = gp_inducing_input_fraction

    @cache
    def gps(self):
        return [
            self.train_gp_model(model)
            for model in new_gp_models(
                [(self.x[x_known], self.reward_functions()[a](self.x[x_known]))
                 for a, x_known in enumerate(self.x_known_on_each_action)],
                gp_inducing_input_fraction=self.gp_inducing_input_fraction)
        ]

    def gp_at_inputs(self, x):
        return [gp.at_inputs(x) for gp in self.gps]

    def reward_function_distributions(self, x):
        return self.gp_at_inputs(x)


class PlateauRewardRealityExperiment(RealityExperiment):
    def __init__(self,
                 plateau_function_distribution,
                 *args,
                 stddev=0.0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.plateau_function_distribution = plateau_function_distribution
        self.stddev = stddev

        for i, f in enumerate(self.plateau_functions):
            self.x_known_on_each_action[i] = f.in_bounds(self.x_train)

        self.x_known_on_each_action = [
            np.concatenate(
                [x_known, np.full([len(self.x_test)], False)], axis=0)
            for x_known in self.x_known_on_each_action
        ]

    def _compute_plateau_functions(self):
        self.reset_random()
        return [
            self.plateau_function_distribution.sample()
            for _ in range(self.num_actions)
        ]

    @cache
    def plateau_functions(self):
        return load_or_save(
            lambda: load_all_plateau_functions(self.id),
            lambda functions: [
                f.save('plateau_function.{}.{}'.format(self.id, a))
                for a, f in enumerate(functions)
            ]
        )(self._compute_plateau_functions)()

    def reward_functions(self):
        def new_r(f):
            def eval_and_expand(x):
                return np.expand_dims(f(x, stddev=self.stddev), axis=1)

            return eval_and_expand

        return list(map(new_r, self.plateau_functions))


class Experiment(object):
    def __init__(self,
                 new_reality_experiment,
                 sample_num_actions,
                 x_train,
                 x_test=None):
        if x_test is None: x_test = x_train

        self.new_reality_experiment = new_reality_experiment
        self.sample_num_actions = sample_num_actions
        tf.train.get_or_create_global_step().assign(0)
        self.x_train = x_train
        self.x_train.setflags(write=False)

        self.x_test = x_test
        self.x_test.setflags(write=False)

        self._realities = {}

    def __getitem__(self, i):
        if i not in self._realities:
            self._realities[i] = self.new_reality_experiment(
                i, self.sample_num_actions(), self.x_train, self.x_test)
        return self._realities[i]
