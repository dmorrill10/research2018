import tensorflow as tf
import numpy as np
from robust_offline_contextual_bandits.competitor import TileCodingCompetitor


class CompetitorTest(tf.test.TestCase):
    def test_tile_coding_competitor(self):
        x = np.expand_dims(np.linspace(-1, 1, 512).astype('float32'), axis=1)
        patient = TileCodingCompetitor.new_rep(128, x)
        self.assertAllClose(512 / (2 * 128 + 1.0), patient.learning_rate())

        tf.train.get_or_create_global_step()
        patient = TileCodingCompetitor(128, x, 1)
        self.assertAllClose(512 / (2 * 128 + 1.0), patient.rep.learning_rate())


if __name__ == '__main__':
    tf.test.main()
