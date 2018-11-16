from collections import namedtuple, UserList, UserDict
import scipy.stats as st
import numpy as np
import tensorflow as tf
import os
try:
    from pydrive.auth import GoogleAuth
    from pydrive.drive import GoogleDrive
    from google.colab import auth
    from oauth2client.client import GoogleCredentials
except:
    pass

DataComponentsForTraining = namedtuple(
    'DataComponentsForTraining',
    ['data', 'noisy_data', 'combined_raw_data', 'sort_indices'])


def mean_and_t_ci(a, axis=-1, confidence=0.95):
    sem = st.sem(a, axis=axis)
    prob_mass = st.t.ppf((1.0 + confidence) / 2.0, a.shape[axis] - 1.0)
    return a.mean(axis=axis), sem * prob_mass


class HomogeneousDataGatherer(object):
    def __init__(self, *dimensions):
        self._data = []
        self._dimensions = dimensions

    def np(self, *dimensions):
        a = np.array(self._data)
        if len(dimensions) > 0:
            a = a.reshape(dimensions)
        elif len(self._dimensions) > 0:
            a = a.reshape(self._dimensions)
        return a

    def tf(self, *dimensions):
        a = tf.stack(self._data)
        if len(dimensions) > 0:
            a = tf.reshape(a, dimensions)
        elif len(self._dimensions) > 0:
            a = tf.reshape(a, self._dimensions)
        return a

    def set_dimensions(self, *dimensions):
        self._dimensions = dimensions

    def append(self, datum):
        self._data.append(datum)


class GoogleDriveWrapper(object):
    def __init__(self):
        auth.authenticate_user()
        self.gauth = GoogleAuth()
        self.gauth.credentials = GoogleCredentials.get_application_default()
        self.drive = GoogleDrive(self.gauth)

    def save(self, path, gd_dir_id=None):
        metadata = {'title': os.path.basename(path)}
        if gd_dir_id is not None:
            metadata['parents'] = [{
                'kind': 'drive#childList',
                'id': gd_dir_id
            }]
        file = self.drive.CreateFile(metadata)
        file.SetContentFile(path)
        file.Upload()
        return self

    def ls(self, gd_dir_id):
        return [
            file_info['title']
            for file_info in self.drive.ListFile({
                'q':
                "'{}' in parents".format(gd_dir_id)
            }).GetList()
        ]

    def load(self, file_name=None, gd_dir_id=None):
        query = {}
        if gd_dir_id is not None:
            query['q'] = "'{}' in parents and trashed=false".format(gd_dir_id)
        if file_name is not None:
            query['orderBy'] = 'modifiedDate desc'
        for file_info in self.drive.ListFile(query).GetList():
            if file_name is None or file_info['title'] == file_name:
                f = self.drive.CreateFile({'id': file_info['id']})
                f.GetContentFile(file_info['title'])
                break
        return self


def load_or_save(load, save):
    def f(compute):
        def g(*args, **kwargs):
            try:
                payload = load()
                tf.logging.info('Loaded data...')
            except:
                tf.logging.info('Computing and saving data...')
                payload = compute(*args, **kwargs)
                save(payload)
            return payload

        return g

    return f


def load_list(load):
    def f(*args, **kwargs):
        l = load(*args, **kwargs)
        if len(l) < 1:
            raise FileNotFoundError('No files found.')
        return l

    return f


class TaggedDatum(UserDict):
    def __init__(self, payload, **tags):
        super().__init__(tags)
        self._payload = payload

    def __missing__(self, key):
        return None

    def __eq__(self, other):
        return self() == other() and UserDict.__eq__(self, other)

    def __call__(self):
        return self._payload

    @property
    def tags(self):
        return self.data

    def with_tags(self, new_payload):
        return self.__class__(new_payload, **self.tags)


class TaggedData(UserList):
    def append(self, item, **tags):
        super().append(TaggedDatum(item, **tags))
