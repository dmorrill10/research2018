import tensorflow as tf
tf.compat.v1.enable_eager_execution()
from research2018.tabular_cfr import TabularCfr, TabularCfrCurrent


class TabularCfrTest(tf.test.TestCase):
    def setUp(self):
        tf.random.set_seed(42)

    def test_zeros(self):
        num_info_sets = 2
        num_actions = 3
        patient = TabularCfr.zeros(num_info_sets, num_actions)
        self.assertAllClose(
            tf.fill([num_info_sets, num_actions], 1.0 / num_actions),
            patient.cur())
        self.assertAllClose(
            tf.fill([num_info_sets, num_actions], 1.0 / num_actions),
            patient.avg())
        self.assertAllClose(
            tf.fill([num_info_sets, num_actions], 1.0 / num_actions),
            patient.policy())

    def test_update(self):
        num_info_sets = 2
        num_actions = 3
        patient = TabularCfr(
            TabularCfrCurrent(tf.random_normal([num_info_sets, num_actions])),
            tf.zeros([num_info_sets, num_actions]))

        initial_cur = tf.constant([[0.50621, 0., 0.49379],
                                   [0.333333, 0.333333, 0.333333]])
        self.assertAllClose(initial_cur, patient.cur())
        self.assertAllClose(
            tf.fill([num_info_sets, num_actions], 1.0 / num_actions),
            patient.avg())
        self.assertAllClose(
            tf.fill([num_info_sets, num_actions], 1.0 / num_actions),
            patient.policy())

        def env(policy):
            return tf.random_normal([num_info_sets, num_actions]) * policy

        patient.update(env)

        next_cur = tf.constant([[0.39514, 0., 0.60486],
                                [0.333333, 0.333333, 0.333333]])
        self.assertAllClose(next_cur, patient.cur())
        self.assertAllClose(initial_cur, patient.avg())
        self.assertAllClose(initial_cur, patient.policy())

        patient.update(env)

        next_next_cur = [[0., 0., 1.], [0.333333, 0.333333, 0.333333]]
        self.assertAllClose(next_next_cur, patient.cur())
        self.assertAllClose((initial_cur + next_cur) / 2.0, patient.avg())
        self.assertAllClose((initial_cur + next_cur) / 2.0, patient.policy())


if __name__ == '__main__':
    tf.test.main()
