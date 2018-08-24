import tensorflow as tf
try:
    tf.enable_eager_execution()
except:
    pass
from robust_offline_contextual_bandits.named_results import NamedResults
import numpy as np


class NamedResultsTest(tf.test.TestCase):
    def setUp(self):
        np.random.seed(42)

    def test_creation(self):
        patient = NamedResults(np.random.normal(size=[10, 3]))
        assert patient.num_evs() == 10
        assert patient.num_reps() == 3
        self.assertAllClose(patient.avg_evs(), [
            0.335379, 0.35158, 0.625724, -0.128862, -1.132079, -0.42029,
            -0.284893, -0.527665, -0.528151, -0.172211
        ])
        self.assertAlmostEqual(patient.min_ev(), -1.132078602)


if __name__ == '__main__':
    tf.test.main()
