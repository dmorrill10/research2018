import tensorflow as tf
from research2018 import mnist


class MnistTest(tf.test.TestCase):
    def test_new_training_dataset_and_normalizer(self):
        for constant_reward in [None, 0.5]:
            mnist.new_training_dataset_and_normalizer(constant_reward)

    def test_subtract_mean_scale_to_zero_one_data_normalizer(self):
        data, normalizer = mnist.new_training_dataset_and_normalizer()
        normalized_x = normalizer(data.x)
        self.assertAlmostEqual(0.0, tf.reduce_mean(normalized_x).numpy())
        self.assertLessEqual(tf.reduce_max(tf.abs(normalized_x)).numpy(), 1.0)


if __name__ == '__main__':
    tf.test.main()
