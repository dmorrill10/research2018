class OnlineLearner(object):
    def __init__(self, learner, train_env, test_env=None):
        self.learner = learner
        self.train_env = train_env
        self.test_env = test_env

    @property
    def policy(self):
        return self.learner.policy

    def update(self, *args, **kwargs):
        return self.learner.update(self.train_env, *args, **kwargs)

    def test_ev(self):
        return self.test_env(self.policy())
