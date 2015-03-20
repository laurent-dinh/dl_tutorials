import numpy
import fuel
from fuel.datasets import IndexableDataset, MNIST
from sklearn.datasets import make_classification

from dl_tutorials.utils.whitening import build_mean_covariance


def build_2d_datasets(dataset_name, n_train=20):
    if dataset_name not in ['mnist', 'sklearn', 'xor']:
        raise ValueError('This dataset is not supported')

    if dataset_name == 'xor':
        data_x = numpy.random.normal(
            size=(5000, 2)
        ).astype(dtype=fuel.config.floatX)
        which_cluster = (numpy.random.uniform(size=(data_x.shape[0], 2)) > .5)
        data_x += 2. * (2 * which_cluster - 1)
        data_y = (2 * which_cluster - 1).prod(axis=1) * .5 + .5
        data_y = data_y.astype(dtype='int32').reshape((-1, 1))
    if dataset_name == 'sklearn':
        data_x, data_y = make_classification(
            n_samples=1000,
            n_features=2,
            n_informative=2,
            n_redundant=0,
            n_classes=2)
        data_y = data_y.astype(dtype='int32').reshape((-1, 1))
    if dataset_name == 'mnist':
        dataset = MNIST('train')
        data_mean, data_cov = build_mean_covariance(dataset, 256)
        eigval, eigvec = numpy.linalg.eigh(data_cov)
        features = (dataset.indexables[0] - data_mean).dot(eigvec[:, -2:])
        features_pos = features[dataset.indexables[1][:, 0] == 3]
        features_neg = features[dataset.indexables[1][:, 0] == 5]

        data_x = numpy.zeros(
            (features_pos.shape[0] + features_neg.shape[0], 2)
        )
        data_x[:n_train] = features_pos[:n_train]
        data_x[n_train:(2 * n_train)] = features_neg[:n_train]
        data_x[(2 * n_train):-(features_neg.shape[0] - n_train)] = \
            features_pos[n_train:]
        data_x[-(features_neg.shape[0] - n_train):] = features_neg[n_train:]

        data_y = numpy.zeros(
            (features_pos.shape[0] + features_neg.shape[0], 1)
        )
        data_y[:n_train] = 1
        data_y[n_train:(2 * n_train)] = 0
        data_y[(2 * n_train):-(features_neg.shape[0] - n_train)] = 1
        data_y[-(features_neg.shape[0] - n_train):] = 0

    train_dataset = IndexableDataset({
        'features': data_x[:(2 * n_train)],
        'targets': data_y[:(2 * n_train)]
    })
    test_dataset = IndexableDataset({
        'features': data_x[(2 * n_train):],
        'targets': data_y[(2 * n_train):]
    })

    return train_dataset, test_dataset
