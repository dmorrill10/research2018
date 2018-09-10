import tensorflow as tf
import numpy as np
from robust_offline_contextual_bandits.representations import \
    RepresentationWithFixedInputs


class RepresentationTest(tf.test.TestCase):
    def test_tile_coding(self):
        x = np.expand_dims(np.linspace(-1, 1, 512).astype('float32'), axis=1)
        patient = RepresentationWithFixedInputs.dense_tile_coding(x, 128)
        self.assertAllClose(512 / (2 * 128 + 1.0), patient.learning_rate())


if __name__ == '__main__':
    tf.test.main()
