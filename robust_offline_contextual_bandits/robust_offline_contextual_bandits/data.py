from collections import namedtuple
import scipy.stats as st

DataComponentsForTraining = namedtuple(
    'DataComponentsForTraining',
    ['data', 'noisy_data', 'combined_raw_data', 'sort_indices'])


def mean_and_t_ci(a, axis=-1, confidence=0.95):
    sem = st.sem(a, axis=axis)
    prob_mass = st.t.ppf((1.0 + confidence) / 2.0, a.shape[axis] - 1.0)
    return a.mean(axis=axis), sem * prob_mass
