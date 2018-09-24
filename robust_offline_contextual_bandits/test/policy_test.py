import tensorflow as tf
import numpy as np
from robust_offline_contextual_bandits.policy import max_robust_policy


class PolicyTest(tf.test.TestCase):
    def test_max_robust_policy(self):
        patient = max_robust_policy(
            [
                np.array([True, False]),
                np.array([False, True]),
                np.array([False, False]),
                np.array([True, True]),
            ],
            np.random.normal(size=[2, 4]))
        self.assertAllEqual([[1.0, 0, 0, 0], [0.0, 0, 0, 1.0]], patient)


if __name__ == '__main__':
    tf.test.main()
