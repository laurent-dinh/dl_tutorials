import os
from collections import OrderedDict
import cPickle

import numpy
from fuel import config
from fuel.datasets import IndexableDataset
from fuel.utils import do_not_pickle_attributes


@do_not_pickle_attributes('indexables')
class IMDB(IndexableDataset):
    """IMDB dataset from the deeplearning tutorial."""

    provides_sources = ('features', 'targets')
    folder = 'imdb'
    filename_data = 'imdb.pkl'
    filename_dict = 'imdb.dict.pkl'

    def __init__(self, **kwargs):

        super(IMDB, self).__init__(
            OrderedDict(zip(self.provides_sources,
                            self._load_imdb())),
            **kwargs
        )

    def load(self):
        self.indexables = [data[self.start:self.stop] for source, data
                           in zip(self.provides_sources, self._load_imdb())
                           if source in self.sources]

    def _load_imdb(self):
        dir_path = os.path.join(config.data_path, self.folder)
        with open(os.path.join(dir_path, self.filename_data), 'r') as f:
            features, targets = cPickle.load(f)
        with open(os.path.join(dir_path, self.filename_dict), 'r') as f:
            self.dict = cPickle.load(f)

        features = numpy.array([numpy.array(s, dtype='int32')
                                for s in features])
        targets = numpy.array(targets, dtype='int32').reshape((-1, 1))

        self.inv_dict = {v: k for k, v in self.dict.items()}

        return (features, targets)
