import tensorflow as tf
try:
    tf.enable_eager_execution()
except:
    pass
from tensorflow.python.ops.resource_variable_ops import ResourceVariable
import numpy as np
from robust_offline_contextual_bandits.rm_optimizer import \
    RmOptimizer


class RmOptimizerTest(tf.test.TestCase):
    def setUp(self):
        np.random.seed(42)
        tf.set_random_seed(42)

    def test_linear_single_output(self):
        num_dimensions = 2
        num_players = 1
        num_examples = 10

        x = np.concatenate(
            [
                np.random.normal(size=[num_examples, num_dimensions - 1]),
                np.ones([num_examples, 1])
            ],
            axis=1).astype('float32')
        y = np.random.normal(size=[num_examples, num_players]).astype('float32')

        w = ResourceVariable(tf.zeros([num_dimensions, num_players]))

        loss = tf.losses.mean_squared_error(y, tf.matmul(x, w))
        optimizer = RmOptimizer(polytope_scales=[1])

        self.assertEqual(0.0, tf.reduce_sum(tf.abs(w)).numpy())
        self.assertAlmostEqual(1.1386044, loss.numpy(), places=7)
        for t in range(50):
            with tf.GradientTape() as tape:
                loss = tf.losses.mean_squared_error(y, tf.matmul(x, w))
            grad = tape.gradient(loss, [w])
            optimizer.apply_gradients(zip(grad, [w]))
            if t > 1:
                self.assertLess(loss.numpy(), 1.1386044)
        self.assertAlmostEqual(0.5071932, loss.numpy(), places=6)

    def test_linear_multiple_outputs(self):
        num_dimensions = 2
        num_players = 5
        num_examples = 10

        x = np.concatenate(
            [
                np.random.normal(size=[num_examples, num_dimensions - 1]),
                np.ones([num_examples, 1])
            ],
            axis=1).astype('float32')
        y = np.random.normal(size=[num_examples, num_players]).astype('float32')

        w = ResourceVariable(tf.zeros([num_dimensions, num_players]))

        loss = tf.losses.mean_squared_error(y, tf.matmul(x, w))
        optimizer = RmOptimizer(polytope_scales=[1])

        self.assertEqual(0.0, tf.reduce_sum(tf.abs(w)).numpy())
        self.assertAlmostEqual(0.86844116, loss.numpy(), places=7)
        for t in range(50):
            with tf.GradientTape() as tape:
                loss = tf.losses.mean_squared_error(y, tf.matmul(x, w))
            grad = tape.gradient(loss, [w])
            optimizer.apply_gradients(zip(grad, [w]))
            if t > 1:
                self.assertLess(loss.numpy(), 0.86844116)
        self.assertAlmostEqual(0.6293505, loss.numpy(), places=6)


if __name__ == '__main__':
    tf.test.main()
