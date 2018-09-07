import tensorflow as tf
import numpy as np
from robust_offline_contextual_bandits.competitor import TileCodingCompetitor


class CompetitorTest(tf.test.TestCase):
    def test_tile_coding_competitor(self):
        x = np.expand_dims(
            np.linspace(-1, 1, 512).astype('float32'), axis=1)
        patient = TileCodingCompetitor.new_rep(128, x)
        assert patient.learning_rate() == 4.0

        tf.train.get_or_create_global_step()
        patient = TileCodingCompetitor(128, x, 1)
        assert patient.rep.learning_rate() == 4.0


if __name__ == '__main__':
    tf.test.main()
