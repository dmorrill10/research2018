import tensorflow as tf
import numpy as np
from robust_offline_contextual_bandits.competitor import Competitor


class CompetitorTest(tf.test.TestCase):
    def test_tile_coding_competitor(self):
        tf.train.get_or_create_global_step()
        x = np.expand_dims(np.linspace(-1, 1, 512).astype('float32'), axis=1)
        patient = Competitor.tile_coding(x, 1, 128)
        self.assertAllClose(512 / (2 * 128 + 1.0), patient.rep.learning_rate())


if __name__ == '__main__':
    tf.test.main()
