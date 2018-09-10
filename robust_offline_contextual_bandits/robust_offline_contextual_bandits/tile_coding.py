import numpy as np


class Indexer:
    def __init__(self):
        self.dictionary = {}

    def __str__(self):
        return str(self.dictionary)

    def __len__(self):
        return len(self.dictionary)

    def __contains__(self, obj):
        return obj in self.dictionary

    def __getitem__(self, obj):
        if obj not in self.dictionary:
            self.dictionary[obj] = len(self)
        return self.dictionary[obj]


class TileCoding(object):
    def __init__(self,
                 tile_widths=(1.0, ),
                 num_tiling_pairs=0,
                 min_position=None,
                 max_position=None):
        self._table = Indexer()
        self.tile_widths = np.array(tile_widths)
        half_tile_widths = self.tile_widths / 2.0

        tiling_shifts = [half_tile_widths]
        for i in range(num_tiling_pairs):
            cumulative_offset = (i + 1) * half_tile_widths / (
                num_tiling_pairs + 1)
            tiling_shifts.append(tiling_shifts[0] + cumulative_offset)
            tiling_shifts.append(tiling_shifts[0] - cumulative_offset)
        self._tiling_shifts = np.array(tiling_shifts)
        self._tiles_for_position = np.concatenate(
            [
                np.expand_dims(
                    np.arange(2 * num_tiling_pairs + 1, dtype=int), axis=1),
                np.zeros(self._tiling_shifts.shape, dtype=int)
            ],
            axis=1)

        self._min_position = min_position
        self._max_position = max_position

    def num_tiles(self, min_position=None, max_position=None):
        if min_position is None:
            min_position = 0.0 if self._min_position is None else self._min_position
        if max_position is None:
            max_position = self._max_position
        return ((np.array(max_position) - np.array(min_position)) /
                self.tile_widths).astype(int) + 1

    def num_features(self, min_position=None, max_position=None):
        return self.num_tilings() * np.prod(
            self.num_tiles(min_position, max_position))

    def num_tilings(self):
        return self._tiling_shifts.shape[0]

    def resolution(self):
        return self.tile_widths / self.num_tilings()

    def on(self, position, min_position=None):
        if min_position is None:
            min_position = 0 if self._min_position is None else self._min_position
        position -= np.array(min_position)
        assert not np.any(position < 0.0)
        self._tiles_for_position[:, 1:] = ((
            position + self._tiling_shifts) / self.tile_widths)

        return [
            self._table[tiling.tobytes()]
            for tiling in self._tiles_for_position
        ]

    def features(self, position, min_position=None, max_position=None):
        x = np.zeros(
            [self.num_features(min_position, max_position)], dtype=int)
        x.put(self.on(position, min_position), 1)
        return x


def tile_coding_dense_feature_expansion(state_dimension_boundaries,
                                        num_tiling_pairs,
                                        tile_width_fractions=None):
    min_position, max_position = zip(*state_dimension_boundaries)

    if tile_width_fractions is None:
        tile_width_fractions = [1.0] * len(min_position)

    state_bucketer = TileCoding(
        tile_widths=((np.array(max_position) - np.array(min_position)) *
                     np.array(tile_width_fractions)),
        num_tiling_pairs=num_tiling_pairs,
        min_position=min_position,
        max_position=max_position)

    def tile_coding_features(state):
        return state_bucketer.features(state)

    learning_rate = 1.0 / (2 * num_tiling_pairs + 1.0)
    return tile_coding_features, state_bucketer.num_features(), learning_rate


def tile_coding_sparse_feature_expansion(state_dimension_boundaries,
                                         num_tiling_pairs,
                                         tile_width_fractions=None):
    min_position, max_position = zip(*state_dimension_boundaries)

    if tile_width_fractions is None:
        tile_width_fractions = [1.0] * len(min_position)

    state_bucketer = TileCoding(
        tile_widths=np.array(state_dimension_boundaries) *
        np.array(tile_width_fractions),
        num_tiling_pairs=num_tiling_pairs,
        min_position=min_position,
        max_position=max_position)

    def tile_coding_features(state):
        return state_bucketer.on(state)

    learning_rate = 1.0 / (2 * num_tiling_pairs + 1.0)
    return tile_coding_features, state_bucketer.num_features(), learning_rate
