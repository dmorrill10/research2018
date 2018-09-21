import tensorflow as tf


def state_idx_map(transitions_dict):
    state_indices = {}
    idx = 0
    for state in transitions_dict.keys():
        state_indices[state] = idx
        idx += 1
    assert len(state_indices) == len(transitions_dict)
    assert idx == len(transitions_dict)
    return state_indices


def test_state_idx_map(transitions):
    _patient = state_idx_map(transitions)
    assert len(_patient) == len(transitions)
    assert len(_patient) == len(transitions)

    max_idx = len(_patient)
    _patient = {
        k: tf.one_hot(idx, depth=max_idx)
        for k, idx in _patient.items()
    }
    assert len(_patient) == len(transitions)


if __name__ == '__main__':
    raise NotImplementedError()
    # test_state_idx_map(transitions)
