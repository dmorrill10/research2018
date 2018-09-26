import tensorflow as tf
try:
    tf.enable_eager_execution()
except:
    pass
import numpy as np
from robust_offline_contextual_bandits.experiment import \
    PlateauRewardRealityExperiment, \
    GpRealityExperimentMixin
from robust_offline_contextual_bandits.plateau_function import \
    PlateauFunctionDistribution
from robust_offline_contextual_bandits.gp import Gp


class GpPlateauRewardRealityExperiment(GpRealityExperimentMixin,
                                       PlateauRewardRealityExperiment):
    def __init__(self, *args, **kwargs):
        super().__init__(Gp.gp_regression, *args, **kwargs)


class RealityExperimentTest(tf.test.TestCase):
    def setUp(self):
        np.random.seed(42)
        tf.set_random_seed(42)

    def test_max_robust_policy(self):
        x_train = np.random.normal(scale=2, size=[3, 1])
        x_test = np.random.normal(scale=2, size=[2, 1])
        num_actions = 3

        patient = PlateauRewardRealityExperiment(
            PlateauFunctionDistribution(
                min(x_train.min(), x_test.min()),
                max(x_train.max(), x_test.max()), 2, 5),
            0,
            num_actions,
            x_train,
            x_test,
            stddev=0.0,
            save_to_disk=False)

        x_known = patient.in_bounds(x_test)
        policy = patient.max_robust_policy(x_test, x_known)

        self.assertAllClose([[1.0, 0, 0], [1.0, 0, 0]], policy)

    def test_map_policy(self):
        x_train = np.random.normal(scale=2, size=[3, 1])
        x_test = np.random.normal(scale=2, size=[2, 1])
        num_actions = 3

        patient = GpPlateauRewardRealityExperiment(
            PlateauFunctionDistribution(
                min(x_train.min(), x_test.min()),
                max(x_train.max(), x_test.max()), 2, 5),
            0,
            num_actions,
            x_train,
            x_test,
            stddev=0.0,
            save_to_disk=False)
        policy = patient.map_policy(x_test)

        self.assertAllClose([[0.0, 0, 1], [0.0, 0, 1]], policy)


if __name__ == '__main__':
    tf.test.main()
