import numpy
from fuel.streams import DataStream
from fuel.transformers import Mapping
from fuel.schemes import SequentialScheme


def build_mean_covariance(dataset, batch_size):
    """Compute the mean and the covariance of a dataset.

    Parameters
    ----------
    dataset : fuel.datasets.Dataset
        Dataset whose mean and covariance is computed.
    batch_size : int
        The batch size for computing these quantities.

    Returns
    -------
    X_mean : :class:`~numpy.ndarray`, shape (dim,)
        The mean of the dataset.
    X_cov : :class:`~numpy.ndarray`, shape (dim, dim)
        The covariance matrix of the dataset.

    """
    data_stream = Mapping(
        data_stream=DataStream(
            dataset,
            iteration_scheme=SequentialScheme(
                examples=dataset.num_examples,
                batch_size=batch_size
            )
        ),
        mapping=lambda x: x[dataset.sources.index('features')],
    )

    dataset_iterator = data_stream.get_epoch_iterator()

    unnormalized_mean = 0.
    unnormalized_dot = 0.

    for data in dataset_iterator:
        unnormalized_dot += data.T.dot(data) / batch_size
        unnormalized_mean += data.sum(axis=0) / batch_size
        del data

    X_mean = unnormalized_mean \
        * (float(batch_size) / dataset.num_examples)
    X_cov = unnormalized_dot \
        * (float(batch_size) / dataset.num_examples) \
        - numpy.outer(X_mean, X_mean)

    return X_mean, X_cov
