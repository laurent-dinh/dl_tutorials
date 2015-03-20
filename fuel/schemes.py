import numpy
from picklable_itertools import imap, _iter
from fuel import config
from fuel.schemes import BatchScheme


class BatchwiseShuffledScheme(BatchScheme):
    """Shuffled batches iterator, sequential inside each minibatch.
    Iterate over all the examples in a dataset of fixed size in shuffled
    batches.
    Notes
    -----
    The batch size isn't enforced, so the last batch could be smaller.
    Shuffling the batches requires creating a shuffled list of indices in
    memory. This can be memory-intensive for very large numbers of examples
    (i.e. in the order of tens of millions).
    """
    def __init__(self, use_slice=False, *args, **kwargs):
        self.rng = kwargs.pop('rng', None)
        if self.rng is None:
            self.rng = numpy.random.RandomState(config.default_seed)
        super(BatchwiseShuffledScheme, self).__init__(*args, **kwargs)

        self.use_slice = use_slice

    def get_request_iterator(self):
        indices = list(self.indices)[::self.batch_size]
        self.rng.shuffle(indices)

        if self.use_slice:
            return imap(slice, _iter(indices),
                        imap(lambda x: x + self.batch_size, _iter(indices)))
        else:
            return imap(range, _iter(indices),
                        imap(lambda x: x + self.batch_size
                             if x != self.indices[-1]
                             - (self.indices[-1] % self.batch_size)
                             else self.indices[-1], _iter(indices)))
