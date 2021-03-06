import tensorflow as tf
tf.compat.v1.enable_eager_execution()
from tensorflow.python.ops.resource_variable_ops import ResourceVariable
import numpy as np
from robust_offline_contextual_bandits.optimizers import CompositeOptimizer
from robust_offline_contextual_bandits import rm_optimizers


class RmOptimizerTest(tf.test.TestCase):
    def setUp(self):
        np.random.seed(42)
        tf.random.set_seed(42)

    def test_l1_mrr_linear_multiple_outputs(self):
        num_dimensions = 2
        num_players = 5
        num_examples = 10

        x = np.concatenate([
            np.random.normal(size=[num_examples, num_dimensions - 1]),
            np.ones([num_examples, 1])
        ],
                           axis=1).astype('float32')
        y = np.random.normal(
            size=[num_examples, num_players]).astype('float32')

        w = ResourceVariable(tf.zeros([num_dimensions, num_players]))

        loss = tf.reduce_mean(tf.keras.losses.mse(y, tf.matmul(x, w)))
        optimizer = CompositeOptimizer(
            lambda var: rm_optimizers.RmL1AmrrVariableOptimizer(var,
                                                                scale=1000.0),
            var_list=[w])

        self.assertEqual(0.0, tf.reduce_sum(tf.abs(w)).numpy())
        self.assertAlmostEqual(0.86844116, loss.numpy(), places=6)
        for t in range(10):
            with tf.GradientTape() as tape:
                loss = tf.reduce_mean(tf.keras.losses.mse(y, tf.matmul(x, w)))
            grad = tape.gradient(loss, [w])
            optimizer.apply_gradients(zip(grad, [w]))
            if t > 1:
                self.assertLess(loss.numpy(), 0.86844116)
        self.assertAlmostEqual(0.85839784, loss.numpy(), places=6)

        # Compare this to rm_optimizers.RmL1VariableOptimizer:
        w = ResourceVariable(tf.zeros([num_dimensions, num_players]))

        loss = tf.reduce_mean(tf.keras.losses.mse(y, tf.matmul(x, w)))
        optimizer = CompositeOptimizer(
            lambda var: rm_optimizers.RmL1VariableOptimizer(var, scale=1000.0),
            var_list=[w])

        self.assertEqual(0.0, tf.reduce_sum(tf.abs(w)).numpy())
        self.assertAlmostEqual(0.86844116, loss.numpy(), places=6)
        for t in range(10):
            with tf.GradientTape() as tape:
                loss = tf.reduce_mean(tf.keras.losses.mse(y, tf.matmul(x, w)))
            grad = tape.gradient(loss, [w])
            optimizer.apply_gradients(zip(grad, [w]))
        self.assertAlmostEqual(97.47674, loss.numpy(), places=4)

    def test_inf_linear_single_output(self):
        num_dimensions = 2
        num_players = 1
        num_examples = 10

        x = np.concatenate([
            np.random.normal(size=[num_examples, num_dimensions - 1]),
            np.ones([num_examples, 1])
        ],
                           axis=1).astype('float32')
        y = np.random.normal(
            size=[num_examples, num_players]).astype('float32')

        w = ResourceVariable(tf.zeros([num_dimensions, num_players]))

        loss = tf.reduce_mean(tf.keras.losses.mse(y, tf.matmul(x, w)))
        optimizer = CompositeOptimizer(
            lambda var: rm_optimizers.RmInfVariableOptimizer(var, scale=0.8))

        self.assertEqual(0.0, tf.reduce_sum(tf.abs(w)).numpy())
        self.assertAlmostEqual(1.1386044, loss.numpy(), places=7)
        for t in range(50):
            with tf.GradientTape() as tape:
                loss = tf.reduce_mean(tf.keras.losses.mse(y, tf.matmul(x, w)))
            grad = tape.gradient(loss, [w])
            optimizer.apply_gradients(zip(grad, [w]))
            if t > 1:
                self.assertLess(loss.numpy(), 1.1386044)
        self.assertAlmostEqual(0.5145497, loss.numpy(), places=6)
        self.assertGreater(tf.reduce_sum(tf.abs(w)), 0.8)

    def test_nn_two_column_weights_only(self):
        num_dimensions = 3
        num_players = 2

        w = ResourceVariable(tf.zeros([num_dimensions, num_players]))

        y = 0.1

        loss = tf.reduce_mean(tf.keras.losses.mse(tf.fill(w.shape, y), w))
        optimizer = CompositeOptimizer(
            lambda var: rm_optimizers.RmNnVariableOptimizer(var, scale=1.0))

        self.assertAlmostEqual(y * y, loss.numpy())
        for i in range(20):
            with tf.GradientTape() as tape:
                loss = tf.reduce_mean(
                    tf.keras.losses.mse(tf.fill(w.shape, y), w))
            grad = tape.gradient(loss, [w])
            optimizer.apply_gradients(zip(grad, [w]))
        loss = tf.reduce_mean(tf.keras.losses.mse(tf.fill(w.shape, y), w))
        self.assertAlmostEqual(0.0, loss.numpy())

    def test_sim_linear_single_output(self):
        num_dimensions = 2
        num_players = 1
        num_examples = 10

        x = np.concatenate([
            np.random.uniform(size=[num_examples, num_dimensions - 1]),
            np.ones([num_examples, 1])
        ],
                           axis=1).astype('float32')
        y = np.random.uniform(
            size=[num_examples, num_players]).astype('float32')

        w = ResourceVariable(tf.zeros([num_dimensions, num_players]))

        loss = tf.reduce_mean(tf.keras.losses.mse(y, tf.matmul(x, w)))
        optimizer = CompositeOptimizer(
            lambda var: rm_optimizers.RmSimVariableOptimizer(var, scale=1))

        self.assertEqual(0.0, tf.reduce_sum(tf.abs(w)).numpy())
        self.assertAlmostEqual(0.23852186, loss.numpy(), places=7)
        for t in range(10):
            with tf.GradientTape() as tape:
                loss = tf.reduce_mean(tf.keras.losses.mse(y, tf.matmul(x, w)))
            grad = tape.gradient(loss, [w])
            optimizer.apply_gradients(zip(grad, [w]))
            if t > 1:
                self.assertLess(loss.numpy(), 0.23852186)
        self.assertAlmostEqual(0.06659414, loss.numpy(), places=6)

    def test_inf_single_column_weights_only(self):
        num_dimensions = 2
        num_players = 1
        max_value = 0.5

        w = ResourceVariable(tf.zeros([num_dimensions, num_players]))

        loss = tf.reduce_mean(w + max_value)
        optimizer = CompositeOptimizer(
            lambda var: rm_optimizers.RmInfVariableOptimizer(var,
                                                             scale=max_value))

        self.assertEqual(max_value, loss.numpy())
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(w + max_value)
        grad = tape.gradient(loss, [w])
        optimizer.apply_gradients(zip(grad, [w]))
        loss = tf.reduce_mean(w + max_value)
        self.assertEqual(0.0, loss.numpy())

    def test_l1_linear_single_output(self):
        num_dimensions = 2
        num_players = 1
        num_examples = 10

        x = np.concatenate([
            np.random.normal(size=[num_examples, num_dimensions - 1]),
            np.ones([num_examples, 1])
        ],
                           axis=1).astype('float32')
        y = np.random.normal(
            size=[num_examples, num_players]).astype('float32')

        w = ResourceVariable(tf.zeros([num_dimensions, num_players]))

        loss = tf.reduce_mean(tf.keras.losses.mse(y, tf.matmul(x, w)))
        optimizer = CompositeOptimizer(
            lambda var: rm_optimizers.RmL1VariableOptimizer(var, scale=1))

        self.assertEqual(0.0, tf.reduce_sum(tf.abs(w)).numpy())
        self.assertAlmostEqual(1.1386044, loss.numpy(), places=7)
        for t in range(50):
            with tf.GradientTape() as tape:
                loss = tf.reduce_mean(tf.keras.losses.mse(y, tf.matmul(x, w)))
            grad = tape.gradient(loss, [w])
            optimizer.apply_gradients(zip(grad, [w]))
            if t > 1:
                self.assertLess(loss.numpy(), 1.1386044)
        self.assertAlmostEqual(0.5071932, loss.numpy(), places=6)

    def test_l1_linear_multiple_outputs(self):
        num_dimensions = 2
        num_players = 5
        num_examples = 10

        x = np.concatenate([
            np.random.normal(size=[num_examples, num_dimensions - 1]),
            np.ones([num_examples, 1])
        ],
                           axis=1).astype('float32')
        y = np.random.normal(
            size=[num_examples, num_players]).astype('float32')

        w = ResourceVariable(tf.zeros([num_dimensions, num_players]))

        loss = tf.reduce_mean(tf.keras.losses.mse(y, tf.matmul(x, w)))
        optimizer = CompositeOptimizer(
            lambda var: rm_optimizers.RmL1VariableOptimizer(var, scale=1.0),
            var_list=[w])

        self.assertEqual(0.0, tf.reduce_sum(tf.abs(w)).numpy())
        self.assertAlmostEqual(0.86844116, loss.numpy(), places=6)
        for t in range(50):
            with tf.GradientTape() as tape:
                loss = tf.reduce_mean(tf.keras.losses.mse(y, tf.matmul(x, w)))
            grad = tape.gradient(loss, [w])
            optimizer.apply_gradients(zip(grad, [w]))
            if t > 1:
                self.assertLess(loss.numpy(), 0.86844116)
        self.assertAlmostEqual(0.6293505, loss.numpy(), places=6)


if __name__ == '__main__':
    tf.test.main()
