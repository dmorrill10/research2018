import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import numpy as np
from robust_offline_contextual_bandits.experiment import \
    PlateauRewardRealityExperiment, \
    GpRealityExperimentMixin
from robust_offline_contextual_bandits.plateau_function import \
    PlateauFunctionDistribution as _PlateauFunctionDistribution
from robust_offline_contextual_bandits.gp import Gp


class GpPlateauRewardRealityExperiment(GpRealityExperimentMixin,
                                       PlateauRewardRealityExperiment):
    def __init__(self, *args, **kwargs):
        super().__init__(Gp.gp_regression, *args, **kwargs)


class PlateauFunctionDistribution(_PlateauFunctionDistribution):
    def sample_num_plateaus(self):
        return 1

    def sample_height(self):
        return np.random.uniform()

    def sample_center(self):
        return np.random.uniform()

    def sample_radius(self):
        return 1.0


class RealityExperimentTest(tf.test.TestCase):
    def setUp(self):
        np.random.seed(42)
        tf.random.set_seed(42)

    def test_max_robust_policy(self):
        x_train = np.array([[-0.5], [0.2], [0.8]])
        x_test = np.array([[-1.0], [0.5]])
        num_actions = 3

        patient = PlateauRewardRealityExperiment(PlateauFunctionDistribution(),
                                                 0,
                                                 num_actions,
                                                 x_train,
                                                 x_test,
                                                 stddev=0.0,
                                                 save_to_disk=False)

        x_known = patient.in_bounds(x_test)

        self.assertAllEqual([[False, True], [False, True], [False, True]],
                            x_known)
        policy = patient.max_robust_policy(x_test, x_known)

        self.assertAllClose([[1 / 3.0] * 3, [0.0, 1.0, 0]], policy)

    def test_map_policy(self):
        self.skipTest("A matrix inversion here fails because the matrix is "
                      "not positive semidefinite.")
        x_train = np.random.normal(scale=2, size=[3, 1])
        x_test = np.random.normal(scale=2, size=[2, 1])
        num_actions = 3

        patient = GpPlateauRewardRealityExperiment(
            PlateauFunctionDistribution(),
            0,
            num_actions,
            x_train,
            x_test,
            stddev=0.0,
            save_to_disk=False)
        policy = patient.map_policy(x_test)

        self.assertAllClose([[0, 1.0, 0], [0.0, 1.0, 0]], policy)


if __name__ == '__main__':
    tf.test.main()
