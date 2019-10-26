import functools
import tensorflow as tf
from robust_offline_contextual_bandits import optimizers
from robust_offline_contextual_bandits import sda_optimizers


class SdaNnOptimizer(optimizers.CompositeOptimizer):
    @classmethod
    def new(cls, num_variables=1, **kwargs):
        def new_vo_fn():
            return functools.partial(
                sda_optimizers.MaxRegretRegularizedSdaInfVariableOptimizer,
                utility_initializer=tf.zeros_initializer(),
                **kwargs)

        return cls.combine(*[new_vo_fn() for _ in range(num_variables)])


class SdaOptimizersTest(tf.test.TestCase):
    def setUp(self):
        tf.random.set_seed(42)

    def test_one_dimension(self):
        y = tf.cast(tf.range(100), tf.float32)

        y_hat = tf.Variable(tf.zeros([1]))

        def loss_fn(my_y, my_y_hat):
            return tf.reduce_mean(tf.keras.losses.mse(my_y, my_y_hat) / 2.0)

        optimizer = sda_optimizers.MaxRegretRegularizedSdaNnVariableOptimizer(
            y_hat, scale=100.0, utility_initializer=tf.zeros_initializer())
        optimizer.dense_update(tf.zeros([]))

        num_updates = 0
        regret_wrt_leader = tf.Variable(tf.zeros([]))
        y_bar = tf.Variable(tf.zeros([1]))
        for y_i in y:
            y_bar = y_bar.assign_add((y_i - y_bar) / (num_updates + 1.0))
            regret_wrt_leader = regret_wrt_leader.assign_add(
                loss_fn(y_i, y_hat) - loss_fn(y_i, y_bar))

            # print(
            #     'y_i: {}, y_hat: {}, y_bar: {}, regret_wrt_leader: {}'.format(
            #         y_i.numpy(), y_hat[0].numpy(), y_bar.numpy(),
            #         regret_wrt_leader.numpy()))

            with tf.GradientTape() as tape:
                loss = loss_fn(y_i, y_hat)
            grad = tape.gradient(loss, [y_hat])
            optimizer.dense_update(grad[0], num_updates)
            num_updates += 1

            # print('avg_utility: {}, avg_ev: {}, regret: {}'.format(
            #     optimizer.avg_utility()[0, 0],
            #     optimizer.avg_ev()[0, 0],
            #     optimizer.regret()[0, 0]))
            # print('')

        self.assertAlmostEqual(-39741, regret_wrt_leader.numpy(), places=0)

    def test_one_dimension_shuffled(self):
        y = tf.random.shuffle(tf.cast(tf.range(100), tf.float32))
        y_hat = tf.Variable(tf.zeros([1]))

        def loss_fn(my_y, my_y_hat):
            return tf.reduce_mean(tf.keras.losses.mse(my_y, my_y_hat) / 2.0)

        optimizer = sda_optimizers.MaxRegretRegularizedSdaNnVariableOptimizer(
            y_hat, scale=100.0, utility_initializer=tf.zeros_initializer())
        optimizer.dense_update(tf.zeros([]))

        num_updates = 0
        regret_wrt_leader = tf.Variable(tf.zeros([]))
        y_bar = tf.Variable(tf.zeros([1]))
        for y_i in y:
            y_bar = y_bar.assign_add((y_i - y_bar) / (num_updates + 1.0))
            regret_wrt_leader = regret_wrt_leader.assign_add(
                loss_fn(y_i, y_hat) - loss_fn(y_i, y_bar))

            # print(
            #     'y_i: {}, y_hat: {}, y_bar: {}, regret_wrt_leader: {}'.format(
            #         y_i.numpy(), y_hat[0].numpy(), y_bar.numpy(),
            #         regret_wrt_leader.numpy()))

            with tf.GradientTape() as tape:
                loss = loss_fn(y_i, y_hat)
            grad = tape.gradient(loss, [y_hat])
            optimizer.dense_update(grad[0], num_updates)
            num_updates += 1

            # print('avg_utility: {}, avg_ev: {}, regret: {}'.format(
            #     optimizer.avg_utility()[0, 0],
            #     optimizer.avg_ev()[0, 0],
            #     optimizer.regret()[0, 0]))
            # print('')

        self.assertAlmostEqual(8183, regret_wrt_leader.numpy(), places=0)

    def test_three_dimension_linear_realizable(self):
        num_dimensions = 3
        num_examples = 100
        num_outputs = 2
        x = tf.random.normal(shape=[num_examples, num_dimensions])
        w_true = tf.random.normal(shape=[num_dimensions, 1])
        b_true = tf.random.normal(shape=[1, num_outputs])
        y = x @ w_true + b_true

        w = tf.Variable(tf.zeros([num_dimensions, num_outputs]))
        b = tf.Variable(tf.zeros([num_outputs]))

        @tf.function
        def y_hat(my_x):
            return my_x @ w + b

        def loss_fn(my_y, my_y_hat):
            return tf.reduce_mean(tf.keras.losses.mse(my_y, my_y_hat) / 2.0)

        optimizer = SdaNnOptimizer.new(num_variables=2, scale=10.)
        optimizer.apply_gradients([(tf.zeros([]), w), (tf.zeros([]), b)])

        self.assertAlmostEqual(0.576, loss_fn(y, y_hat(x)).numpy(), places=3)

        num_updates = 0
        x_losses = [2.1e-5, 2.8e-13]
        for epoch in range(2):
            for x_i, y_i in zip(x, y):
                x_i = tf.expand_dims(x_i, 0)

                with tf.GradientTape() as tape:
                    loss = loss_fn(y_i, y_hat(x_i))
                grad = tape.gradient(loss, [w, b])
                optimizer.apply_gradients(zip(grad, [w, b]))
                num_updates += 1

            self.assertAlmostEqual(x_losses[epoch],
                                   loss_fn(y, y_hat(x)).numpy(),
                                   places=5)

    def test_three_dimension_linear_random(self):
        num_dimensions = 3
        num_examples = 100
        num_outputs = 2
        x = tf.random.normal(shape=[num_examples, num_dimensions])
        y = tf.random.normal(shape=[num_examples, num_outputs])

        w = tf.Variable(tf.zeros([num_dimensions, num_outputs]))
        b = tf.Variable(tf.zeros([num_outputs]))

        @tf.function
        def y_hat(my_x):
            return my_x @ w + b

        def loss_fn(my_y, my_y_hat):
            return tf.reduce_mean(tf.keras.losses.mse(my_y, my_y_hat) / 2.0)

        optimizer = SdaNnOptimizer.new(num_variables=2,
                                       scale=2.,
                                       min_reg_param=1.0)
        optimizer.apply_gradients([(tf.zeros([]), w), (tf.zeros([]), b)])

        self.assertAlmostEqual(0.464, loss_fn(y, y_hat(x)).numpy(), places=3)

        num_updates = 0
        x_losses = [0.446, 0.443]
        for epoch in range(2):
            for x_i, y_i in zip(x, y):
                x_i = tf.expand_dims(x_i, 0)

                with tf.GradientTape() as tape:
                    loss = loss_fn(y_i, y_hat(x_i))
                grad = tape.gradient(loss, [w, b])
                optimizer.apply_gradients(zip(grad, [w, b]))
                num_updates += 1

            # print(loss_fn(y, y_hat(x)).numpy())
            self.assertAlmostEqual(x_losses[epoch],
                                   loss_fn(y, y_hat(x)).numpy(),
                                   places=3)


if __name__ == '__main__':
    tf.test.main()
