from robust_offline_contextual_bandits.plotting import tableu20_color_table


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


class ArmsWithContexts(object):
    def __init__(self, arms, x, stddev=0.0):
        self.arms = arms
        self.x = x
        self.components_for_training = arms.components_for_training(x, stddev)

    def num_contexts(self):
        return len(self.x)

    def num_arms(self):
        return len(self.arms)
