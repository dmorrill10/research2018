import tensorflow as tf
from research2018.rrm import \
    rrm_grad, \
    rm_policy, \
    rrm_utilities, \
    br, \
    behavioral_to_sequence_form_strat
from tensorflow.keras.layers import Dense as DenseLayer


def action_utilities(policy, utility_fn):
    strat = behavioral_to_sequence_form_strat(policy)
    return tf.reshape(utility_fn @ strat, policy.shape)


def update_player(pro_model, opt, contexts, utility_fn, ant_model):
    pro_action_utilities = action_utilities(rm_policy(ant_model(contexts)),
                                            utility_fn)
    opt.apply_gradients(rrm_grad(pro_model, contexts, pro_action_utilities))


def game_utility(p1, p2, utility_fn):
    p1_seq_form_strat = behavioral_to_sequence_form_strat(p1)
    p2_seq_form_strat = behavioral_to_sequence_form_strat(p2)
    return tf.squeeze(
        tf.matmul(p1_seq_form_strat,
                  utility_fn @ p2_seq_form_strat,
                  transpose_a=True))


class ContextualBanditRrmTest(tf.test.TestCase):
    def setUp(self):
        tf.random.set_seed(42)

    def test_static_opponent_independent_predictors_for_each_action_tabular(
            self):
        num_contexts = 10
        num_actions = 2

        utilities = tf.random.normal(shape=[num_contexts, num_actions])
        contexts = tf.eye(num_contexts)

        optimizer = tf.compat.v1.train.GradientDescentOptimizer(
            learning_rate=0.1)
        model = tf.keras.Sequential([
            DenseLayer(num_actions,
                       kernel_initializer=tf.zeros_initializer,
                       use_bias=False)
        ])

        initial_ev = tf.reduce_mean(rrm_utilities(model, contexts, utilities))
        self.assertAllClose(-0.467142, initial_ev)

        optimizer.apply_gradients(rrm_grad(model, contexts, utilities))

        ev = tf.reduce_mean(rrm_utilities(model, contexts, utilities))
        self.assertGreater(ev, initial_ev)
        self.assertAllClose(0.233522, ev)

    def test_static_opponent_independent_predictors_for_each_action(self):
        num_dimensions = 50
        num_contexts = 100
        num_actions = 2

        utilities = tf.random.normal(shape=[num_contexts, num_actions])
        contexts = tf.random.uniform(shape=[num_contexts, num_dimensions])

        optimizer = tf.compat.v1.train.GradientDescentOptimizer(
            learning_rate=0.07)
        model = tf.keras.Sequential([
            DenseLayer(num_actions,
                       kernel_initializer=tf.zeros_initializer,
                       use_bias=False)
        ])

        initial_ev = tf.reduce_mean(rrm_utilities(model, contexts, utilities))
        self.assertAllClose(-0.060263, initial_ev)

        optimizer.apply_gradients(rrm_grad(model, contexts, utilities))

        ev = tf.reduce_mean(rrm_utilities(model, contexts, utilities))
        self.assertGreater(ev, initial_ev)
        self.assertAllClose(-0.053011, ev)

        for t in range(70):
            optimizer.apply_gradients(rrm_grad(model, contexts, utilities))
        self.assertGreater(
            tf.reduce_mean(rrm_utilities(model, contexts, utilities)), 0.15)

    def test_symmetric_independent_predictors_for_each_action_tabular(self):
        num_contexts = 10
        num_actions = 2

        utility_fn = tf.random.normal(
            shape=[num_contexts * num_actions, num_contexts * num_actions])
        contexts = tf.eye(num_contexts)

        p1_model = tf.keras.Sequential([
            DenseLayer(num_actions,
                       kernel_initializer=tf.zeros_initializer,
                       use_bias=False)
        ])
        opt = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1)

        p2_model = tf.keras.Sequential([
            DenseLayer(num_actions,
                       kernel_initializer=tf.zeros_initializer,
                       use_bias=False)
        ])

        def update_p1():
            update_player(p1_model, opt, contexts, utility_fn, p2_model)

        def update_p2():
            update_player(p2_model, opt, contexts, -tf.transpose(utility_fn),
                          p1_model)

        def u(p1, p2):
            return game_utility(p1, p2, utility_fn)

        def br_to_p1():
            return br(
                action_utilities(rm_policy(p1_model(contexts)),
                                 -tf.transpose(utility_fn)))

        def br_to_p2():
            return br(
                action_utilities(rm_policy(p2_model(contexts)), utility_fn))

        p1_policy = rm_policy(p1_model(contexts))
        p2_policy = rm_policy(p2_model(contexts))
        self.assertAllClose(-6.101191, u(p1_policy, p2_policy))
        self.assertAllClose(-20.682821, u(p1_policy, br_to_p1()))
        self.assertAllClose(5.157071, u(br_to_p2(), p2_policy))

        for _ in range(100):
            update_p1()
            update_p2()

        p1_policy = rm_policy(p1_model(contexts))
        p2_policy = rm_policy(p2_model(contexts))
        self.assertAllClose(-7.06723, u(p1_policy, p2_policy))
        self.assertAllClose(-7.158075, u(p1_policy, br_to_p1()))
        self.assertAllClose(-6.82905, u(br_to_p2(), p2_policy))

    def test_symmetric_independent_predictors_for_each_action_random_features(
            self):
        num_dimensions = 80
        num_contexts = 100
        num_actions = 2

        utility_fn = tf.random.normal(
            shape=[num_contexts * num_actions, num_contexts * num_actions])
        contexts = tf.random.uniform(shape=[num_contexts, num_dimensions])

        p1_model = tf.keras.Sequential([
            DenseLayer(num_actions,
                       kernel_initializer=tf.zeros_initializer,
                       use_bias=False)
        ])
        opt = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)

        p2_model = tf.keras.Sequential([
            DenseLayer(num_actions,
                       kernel_initializer=tf.zeros_initializer,
                       use_bias=False)
        ])

        def update_p1():
            update_player(p1_model, opt, contexts, utility_fn, p2_model)

        def update_p2():
            update_player(p2_model, opt, contexts, -tf.transpose(utility_fn),
                          p1_model)

        def u(p1, p2):
            return game_utility(p1, p2, utility_fn)

        def br_to_p1():
            return br(
                action_utilities(rm_policy(p1_model(contexts)),
                                 -tf.transpose(utility_fn)))

        def br_to_p2():
            return br(
                action_utilities(rm_policy(p2_model(contexts)), utility_fn))

        initial_p1_policy = rm_policy(p1_model(contexts))
        initial_p2_policy = rm_policy(p2_model(contexts))
        initial_ev = u(initial_p1_policy, initial_p2_policy)
        self.assertAllClose(69.4488, initial_ev)
        self.assertAllClose(-379.790588, u(initial_p1_policy, br_to_p1()))
        self.assertAllClose(447.33807, u(br_to_p2(), initial_p2_policy))

        for _ in range(100):
            update_p1()
            update_p2()

        p1_policy = rm_policy(p1_model(contexts))
        p2_policy = rm_policy(p2_model(contexts))

        self.assertAllClose(-107.28058, u(p1_policy, br_to_p1()))
        self.assertAllClose(199.11723, u(br_to_p2(), p2_policy))
        self.assertAllClose(106.15553, u(p1_policy, p2_policy))

    def test_br_independent_predictors_for_each_action_random_features(self):
        num_dimensions = 80
        num_contexts = 100
        num_actions = 2

        utility_fn = tf.random.normal(
            shape=[num_contexts * num_actions, num_contexts * num_actions])
        contexts = tf.random.uniform(shape=[num_contexts, num_dimensions])

        model = tf.keras.Sequential([
            DenseLayer(num_actions,
                       kernel_initializer=tf.zeros_initializer,
                       use_bias=False)
        ])
        opt = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.001)

        def br_to_p1(contexts):
            return br(
                action_utilities(rm_policy(model(contexts)),
                                 -tf.transpose(utility_fn)))

        def update_p1():
            update_player(model, opt, contexts, utility_fn, br_to_p1)

        def u():
            return game_utility(rm_policy(model(contexts)), br_to_p1(contexts),
                                utility_fn)

        initial_ev = u()
        self.assertAllClose(-379.790588, initial_ev)
        for t in range(100):
            update_p1()
        self.assertAllClose(-181.28659, u())


if __name__ == '__main__':
    tf.test.main()
