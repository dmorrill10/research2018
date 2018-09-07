import tensorflow as tf
import numpy as np
from robust_offline_contextual_bandits.tile_coding import TileCoding


class TileCodingTest(tf.test.TestCase):
    def test_single_tiling(self):
        for width in np.linspace(0.1, 1.0, 10):
            patient = TileCoding(tile_widths=(width, ))
            self.assertAllClose([width], patient.resolution())
            self.assertAllClose([0], patient.on(0.0 * width))
            self.assertAllClose([0], patient.on(0.25 * width))
            self.assertAllClose([1], patient.on(0.5 * width))
            self.assertAllClose([1], patient.on(0.75 * width))
            self.assertAllClose([1], patient.on(1.0 * width))
            assert patient.num_features(max_position=1.0 * width) == 2

    def test_one_tiling_pair(self):
        for width in np.linspace(0.1, 1.0, 10):
            patient = TileCoding(tile_widths=(width, ), num_tiling_pairs=1)
            self.assertAllClose([width / 3.0], patient.resolution())
            self.assertAllClose([0, 1, 2], patient.on(0.0 * width))
            self.assertAllClose([0, 3, 2], patient.on(0.25 * width))
            self.assertAllClose([4, 3, 2], patient.on(0.5 * width))
            self.assertAllClose([4, 3, 5], patient.on(0.75 * width))
            self.assertAllClose([4, 3, 5], patient.on(1.0 * width))
            assert patient.num_features(max_position=1.0 * width) == 6

    def test_two_tiling_pairs(self):
        for width in np.linspace(0.1, 1.0, 10):
            patient = TileCoding(tile_widths=(width, ), num_tiling_pairs=2)
            self.assertAllClose([width / 5.0], patient.resolution())
            self.assertAllClose([0, 1, 2, 3, 4], patient.on(0.0 * width))
            self.assertAllClose([0, 1, 2, 5, 4], patient.on(0.25 * width))
            self.assertAllClose([6, 7, 2, 5, 4], patient.on(0.5 * width))
            self.assertAllClose([6, 7, 8, 5, 4], patient.on(0.75 * width))
            self.assertAllClose([6, 7, 8, 5, 9], patient.on(1.0 * width))
            assert patient.num_features(max_position=1.0 * width) == 10

    def test_features(self):
        patient = TileCoding(
            tile_widths=(2.0 / 511, ), min_position=-1, max_position=1)
        assert max(patient.on(x / 512 - 1)[0] for x in range(1025)) == 511
        self.assertAllEqual([2.0 / 511], patient.resolution())
        self.assertAllEqual([512], patient.num_tiles())


if __name__ == '__main__':
    tf.test.main()
