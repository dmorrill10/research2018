import numpy as np


class IHT:
    """
    Structure to handle collisions taken from:

    Tile Coding Software version 3.0beta
    by Rich Sutton
    based on a program created by Steph Schaeffer and others
    """

    def __init__(self, sizeval):
        self.size = sizeval
        self.overfullCount = 0
        self.dictionary = {}

    def __str__(self):
        """Prepares a string for printing whenever this object is printed"""
        return "Collision table:" + \
               " size:" + str(self.size) + \
               " overfullCount:" + str(self.overfullCount) + \
               " dictionary:" + str(len(self.dictionary)) + " items"

    def count(self):
        return len(self.dictionary)

    def fullp(self):
        return len(self.dictionary) >= self.size

    def getindex(self, obj, readonly=False):
        d = self.dictionary
        if obj in d:
            return d[obj]
        elif readonly:
            return None
        size = self.size
        count = self.count()
        if count >= size:
            if self.overfullCount == 0:
                print('IHT full, starting to allow collisions')
            self.overfullCount += 1
            return hash(obj) % self.size
        else:
            d[obj] = count
            return count


class TileCoding(object):
    def __init__(self):
        self._cache = {}

    def on_tiles(self, ihtORsize, numtilings, floats, *actions):
        '''
        Many of the computations in tile3's tile depends only on the number of
        tilings. These values can be cached to speedup feature retrieval when
        particular numbers of tilings are used frequently. Converting a numpy
        array to a tuple is expensive though, but converting to a byte string
        representation is fast. The state and action portions of the hash
        keys can then be computed separately and combined just before retrieval.

        For the first 5 200-run episodes in Q1 P2, using tiles from tiles3
        takes about 58.76 seconds, or 11.75 s/run. This method completes the same
        runs in 45.12 s, or 9.024 s/run, thereby providing a 30% speedup.
        '''
        if numtilings not in self._cache:
            self._cache[numtilings] = {}
            a = np.arange(numtilings)
            a = a.reshape([numtilings, 1])
            tilingX2 = (2 * a).repeat(len(floats) - 1, axis=1)

            b = (np.hstack([a, tilingX2]).cumsum(axis=1) + 1e-6) / numtilings
            adjusted_floats = np.hstack([a, b.copy()])
            self._cache[numtilings] = (b, adjusted_floats,
                                       adjusted_floats.view(int))
        b, adjusted_floats, truncated_adjusted_floats = self._cache[numtilings]
        np.add(b, floats, out=adjusted_floats[:, 1:])

        # In-place conversion to ints
        truncated_adjusted_floats[:, 1:] = adjusted_floats[:, 1:]

        state_coord_keys = [
            truncated_adjusted_floats[tiling, :].tobytes()
            for tiling in range(numtilings)
        ]
        tiles = []
        for action in actions:
            action_key = bytes(action)
            tiles.append([
                ihtORsize.getindex(state_key + action_key, False)
                for state_key in state_coord_keys
            ])
        return tiles


class TileCodingRepresentation(object):
    def __init__(self,
                 tile_coding,
                 state_dimension_boundaries,
                 num_tilings,
                 num_tiles,
                 memory_size=4096,
                 num_actions=1):
        self.tile_coding = tile_coding
        self.num_actions = num_actions
        self.num_tiles = np.array(num_tiles, dtype=int)
        self.num_tilings = num_tilings
        self._hash_table = IHT(memory_size)
        self._num_features = (
            self.num_actions * self.num_tilings * np.prod(self.num_tiles))
        state_dimension_boundaries = np.array(state_dimension_boundaries)
        self._state_dimension_offsets = state_dimension_boundaries[:, 0]
        self.space_widths = (
            state_dimension_boundaries[:, 1] - self._state_dimension_offsets)
        tile_width = self.space_widths / num_tiles
        insensitivity_width = tile_width / num_tilings
        self.space_widths += insensitivity_width / 2.0
        self._state_dimension_scalings = self.num_tiles / self.space_widths

    def num_features(self):
        return self._num_features

    def on(self, state, *actions):
        return self.tile_coding.on_tiles(
            self._hash_table, self.num_tilings,
            ((state - self._state_dimension_offsets) *
             self._state_dimension_scalings), *actions)

    def features_from_on(self, on):
        x = np.zeros([self.num_features()], dtype=int)
        x.put(on, 1)
        return x


def tile_coding_dense_feature_expansion(state_dimension_boundaries,
                                        num_tilings, num_tiles):
    state_bucketer = TileCodingRepresentation(
        TileCoding(),
        state_dimension_boundaries=state_dimension_boundaries,
        num_tilings=num_tilings,
        num_tiles=num_tiles,
        memory_size=4 * 410 * num_tilings)

    def tile_coding_features(state):
        on = state_bucketer.on(state, 0)
        assert len(on) == 1
        return state_bucketer.features_from_on(on[0])

    return tile_coding_features, state_bucketer.num_features()


def tile_coding_sparse_feature_expansion(state_dimension_boundaries,
                                         num_tilings, num_tiles):
    state_bucketer = TileCodingRepresentation(
        TileCoding(),
        state_dimension_boundaries=state_dimension_boundaries,
        num_tilings=num_tilings,
        num_tiles=num_tiles,
        memory_size=4 * 410 * num_tilings)

    def tile_coding_features(state):
        return state_bucketer.on(state, 0)[0]

    return tile_coding_features, state_bucketer.num_features()
