import tensorflow as tf
import numpy as np
from robust_offline_contextual_bandits.data import load_or_save, load_list
from robust_offline_contextual_bandits import cache
from robust_offline_contextual_bandits.tf_np import reset_random
from robust_offline_contextual_bandits.plotting import tableu20_color_table
from robust_offline_contextual_bandits.policy import \
    max_robust_policy, \
    greedy_policy
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
                 x_train_known_on_each_action=None):
        if x_train_known_on_each_action is None:
            x_train_known_on_each_action = [
                np.full([len(x_train)], True) for _ in range(num_actions)
            ]
        self.id = id
        self.num_actions = num_actions
        self.x_train = x_train
        self.x_test = x_test

        self.x_train_known_on_each_action = x_train_known_on_each_action
        for xk in self.x_train_known_on_each_action:
            assert len(xk) == len(self.x_train)

    @cache
    def x(self):
        return np.concatenate([self.x_train, self.x_test], axis=0)

    @cache
    def x_test_known_on_each_action(self):
        return [
            np.full([len(self.x_test)], False)
            for _ in range(self.num_actions)
        ]

    @cache
    def x_known_on_each_action(self):
        return [
            np.concatenate(
                [x_known, np.full([len(self.x_test)], False)], axis=0)
            for x_known in self.x_train_known_on_each_action
        ]

    @cache
    def num_training_examples(self):
        return self.x_train_known_on_each_action.sum(axis=0)

    def reset_random(self):
        reset_random(self.id)

    def reward_functions(self):
        raise NotImplementedError('Override')

    def action_colors(self):
        color_table = tableu20_color_table()
        return [next(color_table) for _ in range(self.num_actions)]

    def max_robust_policy(self, x, x_known_on_each_action):
        return max_robust_policy(x_known_on_each_action,
                                 [r(x) for r in self.reward_functions()])

    def map_policy(self, x):
        return greedy_policy(self.avg_rewards(x))

    def reward_sampler(self, x):
        rfds = self.reward_function_distributions(x)

        def sample_rewards(n):
            return tf.transpose(
                tf.stack([action_model(n) for action_model in rfds]),
                [1, 0, 2])

        return sample_rewards

    def reward_function_distributions(self, x):
        raise NotImplementedError('Override')

    def avg_rewards(self, x):
        raise NotImplementedError('Override')


class GpRealityExperimentMixin(object):
    def __init__(self, new_gp_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.new_gp_model = new_gp_model

    @cache
    def gps(self):
        return [
            self.new_gp_model(self.x_train[x_known],
                              self.reward_functions()[a](
                                  self.x_train[x_known])).train()
            for a, x_known in enumerate(self.x_train_known_on_each_action)
        ]

    def reward_function_distributions(self, x):
        return [gp.at_inputs(x) for gp in self.gps]

    def avg_rewards(self, x):
        return np.concatenate(
            [gp.at_inputs(x).mean for gp in self.gps], axis=1)


class PlateauRewardRealityExperiment(RealityExperiment):
    def __init__(self,
                 plateau_function_distribution,
                 *args,
                 stddev=0.0,
                 save_to_disk=True,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._num_plateau_functions = self.num_actions
        self.plateau_function_distribution = plateau_function_distribution
        self.stddev = stddev
        self.save_to_disk = save_to_disk

        self._valid_actions = []
        self.x_train_known_on_each_action = []
        for a, f in enumerate(self.plateau_functions):
            xk = f.in_bounds(self.x_train)
            if xk.sum() > 0:
                self._valid_actions.append(a)
                self.x_train_known_on_each_action.append(xk)
        self.num_actions = len(self._valid_actions)
        self._valid_actions = set(self._valid_actions)

    def in_bounds(self, x):
        return [
            f.in_bounds(x) for a, f in enumerate(self.plateau_functions)
            if a in self._valid_actions
        ]

    def _compute_plateau_functions(self):
        self.reset_random()
        return [
            self.plateau_function_distribution.sample()
            for _ in range(self._num_plateau_functions)
        ]

    @cache
    def plateau_functions(self):
        if self.save_to_disk:

            def save(functions):
                return [
                    f.save('plateau_function.{}.{}'.format(self.id, a))
                    for a, f in enumerate(functions)
                ]
        else:

            def save(functions):
                return None

        return load_or_save(lambda: load_all_plateau_functions(self.id),
                            save)(self._compute_plateau_functions)()

    def reward_functions(self):
        def new_r(f):
            def eval_and_expand(x):
                return np.expand_dims(f(x, stddev=self.stddev), axis=1)

            return eval_and_expand

        return list(
            map(new_r, [
                f for a, f in enumerate(self.plateau_functions)
                if a in self._valid_actions
            ]))


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
