import functools
import tensorflow as tf
from robust_offline_contextual_bandits import optimizers


class MySdaOptimizer(optimizers.CompositeOptimizer):
    @classmethod
    def new(cls, num_layers, scale=1.0):
        def new_vo_fn():
            return functools.partial(
                optimizers.MaxRegretRegularizedSdaNnVariableOptimizer,
                scale=scale,
                utility_initializer=tf.zeros_initializer())

        return cls.combine(*[new_vo_fn() for layer in range(num_layers)])


class MaxRegretRegularizedSdaTest(tf.test.TestCase):
    def setUp(self):
        tf.random.set_seed(42)

    def test_one_dimension(self):
        y = tf.cast(tf.range(100), tf.float32)

        y_hat = tf.Variable(tf.zeros([1]))

        def loss_fn(my_y, my_y_hat):
            return tf.reduce_mean(tf.keras.losses.mse(my_y, my_y_hat) / 2.0)

        optimizer = optimizers.MaxRegretRegularizedSdaNnVariableOptimizer(
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

        optimizer = optimizers.MaxRegretRegularizedSdaNnVariableOptimizer(
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


if __name__ == '__main__':
    tf.test.main()
