import os
from collections import OrderedDict

import numpy
from fuel import config
from fuel.datasets import IndexableDataset
from fuel.utils import do_not_pickle_attributes


@do_not_pickle_attributes('indexables')
class TextFile(IndexableDataset):
    """Take a test file and make it a dataset.

    Parameters
    ----------
    text_file : str
        The path to the text file to change into a dataset.
    """
    provides_sources = ('features',)

    def __init__(self, text_file, **kwargs):
        self.text_file = text_file

        super(TextFile, self).__init__(
            OrderedDict(zip(self.provides_sources,
                            self._load_text())),
            **kwargs
        )

    def load(self):
        self.indexables = [data[self.start:self.stop] for source, data
                           in zip(self.provides_sources, self._load_text())
                           if source in self.sources]

    def _load_text(self):
        if self.text_file.startswith('/'):
            text_path = self.text_file
        else:
            text_path = os.path.join(config.data_path, self.text_file)
        with open(text_path, 'r') as f:
            lines = f.readlines()
        lines_sentences = [unicode(x, errors='ignore').strip().split('. ')
                           for x in lines]
        sentences = reduce(lambda x, y: x + y, lines_sentences)

        self.dict = {'<UNK>': 0}
        idx = 1
        for sentence in sentences:
            for character in sentence:
                if character not in self.dict.keys():
                    self.dict[character] = idx
                    idx += 1
        self.inv_dict = {v: k for k, v in self.dict.items()}

        indexed_sentences = numpy.array(
            [numpy.array([self.dict[c] for c in s], dtype='int32')
             for s in [seq for seq in sentences if len(seq) > 2]]
        )

        return (indexed_sentences,)
