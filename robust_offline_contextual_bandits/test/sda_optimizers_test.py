import functools
import logging
import tensorflow as tf
from robust_offline_contextual_bandits import optimizers
from robust_offline_contextual_bandits import sda_optimizers


class SdaInfOptimizer(optimizers.CompositeOptimizer):
    @classmethod
    def new(cls, num_variables=1, **kwargs):
        def new_vo_fn():
            return functools.partial(
                sda_optimizers.MaxRegretRegularizedSdaInfVariableOptimizer,
                **kwargs)

        return cls.combine(*[new_vo_fn() for _ in range(num_variables)])


class SdaNnOptimizer(optimizers.CompositeOptimizer):
    @classmethod
    def new(cls, num_variables=1, **kwargs):
        def new_vo_fn():
            return functools.partial(
                sda_optimizers.MaxRegretRegularizedSdaNnVariableOptimizer,
                **kwargs)

        return cls.combine(*[new_vo_fn() for _ in range(num_variables)])


class SdaL2Optimizer(optimizers.CompositeOptimizer):
    @classmethod
    def new(cls, num_variables=1, **kwargs):
        def new_vo_fn():
            return functools.partial(
                sda_optimizers.MaxRegretRegularizedSdaL2VariableOptimizer,
                **kwargs)

        return cls.combine(*[new_vo_fn() for _ in range(num_variables)])


@tf.function
def mse(my_y, my_y_hat):
    return tf.reduce_mean(tf.keras.losses.mse(my_y, my_y_hat) / 2.0)


class SdaOptimizersTest(tf.test.TestCase):
    def setUp(self):
        tf.random.set_seed(42)

    def test_mse(self):
        self.assertAlmostEqual(2., mse([2.], [4.]).numpy())
        for y_i in tf.cast(tf.range(100), tf.float32):
            self.assertAlmostEqual(0., mse([y_i], [y_i]).numpy())

    def test_one_dimension(self):
        logging.debug('test_one_dimension')
        y = tf.cast(tf.range(100), tf.float32)

        for new_optimizer in [
                sda_optimizers.MaxRegretRegularizedSdaInfVariableOptimizer,
                sda_optimizers.MaxRegretRegularizedSdaNnVariableOptimizer,
                sda_optimizers.MaxRegretRegularizedSdaL2VariableOptimizer,
        ]:
            logging.debug('Testing optimizer {}.'.format(new_optimizer))
            y_hat = tf.Variable(tf.zeros([1]))

            optimizer = new_optimizer(y_hat, scale=100.)

            num_updates = 0
            regret_wrt_leader = tf.Variable(tf.zeros([]))
            y_bar = tf.Variable(tf.zeros([1]))
            for y_i in y:
                y_bar.assign_add((y_i - y_bar) / (num_updates + 1.0))
                regret_wrt_leader = regret_wrt_leader.assign_add(
                    mse(y_i, y_hat.value()) - mse(y_i, y_bar.value()))

                logging.debug(
                    'y_i: {}, y_hat: {}, y_bar: {}, regret_wrt_leader: {}'.
                    format(y_i.numpy(), y_hat[0].numpy(), y_bar.numpy(),
                           regret_wrt_leader.numpy()))

                with tf.GradientTape() as tape:
                    loss = mse(y_i, y_hat.value())
                grad = tape.gradient(loss, [y_hat])
                optimizer.dense_update(grad[0], num_updates)
                num_updates += 1

                logging.debug('avg_utility: {}, avg_ev: {}, regret: {}'.format(
                    optimizer.avg_utility()[0, 0],
                    optimizer.avg_ev()[0, 0],
                    optimizer.regret()[0, 0]))

            self.assertLess(regret_wrt_leader.numpy(), -33967)

    def test_one_dimension_shuffled(self):
        logging.debug('test_one_dimension_shuffled')
        y = tf.random.shuffle(tf.cast(tf.range(100), tf.float32))

        for new_optimizer in [
                sda_optimizers.MaxRegretRegularizedSdaInfVariableOptimizer,
                sda_optimizers.MaxRegretRegularizedSdaNnVariableOptimizer,
                sda_optimizers.MaxRegretRegularizedSdaL2VariableOptimizer,
        ]:
            logging.debug('Testing optimizer {}.'.format(new_optimizer))
            y_hat = tf.Variable(tf.zeros([1]))

            optimizer = new_optimizer(y_hat, scale=100.0)

            num_updates = 0
            regret_wrt_leader = tf.Variable(tf.zeros([]))
            y_bar = tf.Variable(tf.zeros([1]))
            for y_i in y:
                y_bar.assign_add((y_i - y_bar) / (num_updates + 1.0))
                regret_wrt_leader = regret_wrt_leader.assign_add(
                    mse(y_i, y_hat.value()) - mse(y_i, y_bar.value()))

                logging.debug(
                    'y_i: {}, y_hat: {}, y_bar: {}, regret_wrt_leader: {}'.
                    format(y_i.numpy(), y_hat[0].numpy(), y_bar.numpy(),
                           regret_wrt_leader.numpy()))

                with tf.GradientTape() as tape:
                    loss = mse(y_i, y_hat.value())
                grad = tape.gradient(loss, [y_hat])
                optimizer.dense_update(grad[0], num_updates)
                num_updates += 1

                logging.debug('avg_utility: {}, avg_ev: {}, regret: {}'.format(
                    optimizer.avg_utility()[0, 0],
                    optimizer.avg_ev()[0, 0],
                    optimizer.regret()[0, 0]))

            self.assertLess(regret_wrt_leader.numpy(), 14341)

    def test_three_dimension_linear_realizable(self):
        logging.debug('test_three_dimension_linear_realizable')
        num_dimensions = 3
        num_examples = 100
        num_outputs = 2
        x = tf.random.normal(shape=[num_examples, num_dimensions])
        w_true = tf.abs(tf.random.normal(shape=[num_dimensions, 1]))
        b_true = tf.abs(tf.random.normal(shape=[1, num_outputs]))
        y = x @ w_true + b_true
        max_scale = tf.reduce_sum(w_true)

        for new_optimizer in [
                SdaInfOptimizer.new, SdaNnOptimizer.new, SdaL2Optimizer.new
        ]:
            logging.debug('Testing optimizer {}'.format(new_optimizer))
            w = tf.Variable(tf.zeros([num_dimensions, num_outputs]))
            b = tf.Variable(tf.zeros([num_outputs]))

            @tf.function
            def y_hat(my_x):
                return my_x @ w + b

            optimizer = new_optimizer(num_variables=2, scale=max_scale)

            self.assertAlmostEqual(0.547, mse(y, y_hat(x)).numpy(), places=3)

            num_updates = 0
            x_losses = [2.1e-5, 2.8e-13]
            for epoch in range(2):
                for x_i, y_i in zip(x, y):
                    x_i = tf.expand_dims(x_i, 0)

                    with tf.GradientTape() as tape:
                        loss = mse(y_i, y_hat(x_i))
                    grad = tape.gradient(loss, [w, b])
                    optimizer.apply_gradients(zip(grad, [w, b]))
                    num_updates += 1

                self.assertLess(mse(y, y_hat(x)).numpy(), x_losses[epoch])

    def test_three_dimension_linear_random(self):
        logging.debug('test_three_dimension_linear_random')
        num_dimensions = 16
        num_examples = 100
        num_outputs = 2
        x = tf.random.normal(shape=[num_examples, num_dimensions])
        y = tf.random.normal(shape=[num_examples, num_outputs])

        for new_optimizer in [
                SdaInfOptimizer.new, SdaNnOptimizer.new, SdaL2Optimizer.new
        ]:
            logging.debug('Testing optimizer {}'.format(new_optimizer))

            w = tf.Variable(tf.zeros([num_dimensions, num_outputs]))
            b = tf.Variable(tf.zeros([num_outputs]))

            @tf.function
            def y_hat(my_x):
                return my_x @ w + b

            optimizer = new_optimizer(num_variables=2, scale=1.)

            self.assertAlmostEqual(0.464, mse(y, y_hat(x)).numpy(), places=3)

            num_updates = 0
            x_losses = [0.7, 0.56, 0.51, 0.49, 0.47, 0.46]
            for epoch in range(6):
                for x_i, y_i in zip(x, y):
                    x_i = tf.expand_dims(x_i, 0)

                    with tf.GradientTape() as tape:
                        loss = mse(y_i, y_hat(x_i))
                    grad = tape.gradient(loss, [w, b])
                    optimizer.apply_gradients(zip(grad, [w, b]))
                    num_updates += 1

                my_loss = mse(y, y_hat(x)).numpy()
                logging.debug(my_loss)
                self.assertLess(my_loss, x_losses[epoch])


if __name__ == '__main__':
    tf.test.main()
