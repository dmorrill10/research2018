import numpy as np


class KofnCfr(object):
    '''k-of-n mixin designed to override a `FixedParameterCfr` class.'''

    def __init__(self, opponent, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.opponent = opponent

    def params(self, sess=None):
        params = super().params()
        opponent = self.opponent if sess is None else sess.run(self.opponent)
        params['opponent'] = opponent
        return params

    def graph_save(self, name, sess):
        np.save('{}.params'.format(name), self.params(sess))
        self.cfr.graph_save('{}.cfr'.format(name), sess)
        return self
