#!/usr/bin/env python

import logging
from argparse import ArgumentParser

from theano import tensor
from blocks.algorithms import GradientDescent, Momentum
from blocks.model import Model
from blocks.extensions import FinishAfter, Timing, Printing
from blocks.extensions.monitoring import (DataStreamMonitoring,
                                          TrainingDataMonitoring)
from blocks.main_loop import MainLoop
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme, ShuffledScheme
from fuel.datasets import MNIST
from fuel.transformers import ForceFloatX

from dl_tutorials.part_2_mlp.softmax_regression import SoftmaxRegressor


class TupleMapping(object):
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, args):
        return (self.fn(args[0]), )


def main(num_epochs=1000):
    # MODEL
    # defining the data
    x = tensor.matrix('features')
    y = tensor.lmatrix('targets')

    # defining the model
    softmax_regressor = SoftmaxRegressor(input_dim=784, n_classes=10)

    # defining the cost to learn on
    probs = softmax_regressor.get_probs(features=x)
    cost = softmax_regressor.get_cost(probs=probs, targets=y).mean()
    cost.name = 'cost'

    # defining the cost to monitor
    misclassification = softmax_regressor.get_misclassification(
        probs=probs, targets=y
    ).mean()
    misclassification.name = 'misclassification'

    # DATASETS
    # defining the datasets
    train_dataset = MNIST('train')
    test_dataset = MNIST('test')

    # TRAINING ALGORITHM
    # defining the algorithm
    params = softmax_regressor.get_params()
    algorithm = GradientDescent(
        cost=cost,
        params=params,
        step_rule=Momentum(learning_rate=0.1,
                           momentum=0.1))

    # defining the data stream
    # how the dataset is read
    train_data_stream = ForceFloatX(
        data_stream=DataStream(
            dataset=train_dataset,
            iteration_scheme=ShuffledScheme(
                examples=train_dataset.num_examples,
                batch_size=100,
            )
        )
    )
    test_data_stream = ForceFloatX(
        data_stream=DataStream(
            dataset=test_dataset,
            iteration_scheme=SequentialScheme(
                examples=test_dataset.num_examples,
                batch_size=1000,
            )
        )
    )

    # MONITORING
    # defining the extensions
    extensions = []
    # timing the training and each epoch
    extensions.append(Timing())
    # ending the training after a certain number of epochs
    extensions.append(FinishAfter(after_n_epochs=num_epochs))
    # monitoring the test set
    extensions.append(DataStreamMonitoring(
        [cost, misclassification],
        test_data_stream,
        prefix='test'))
    # monitoring the training set while training
    extensions.append(TrainingDataMonitoring(
        [cost, misclassification],
        prefix='train',
        after_every_epoch=True))
    # printing quantities
    extensions.append(Printing())

    # MERGING IT TOGETHER
    # defining the model
    model = Model(cost)

    # defining the training main loop
    main_loop = MainLoop(model=model,
                         data_stream=train_data_stream,
                         algorithm=algorithm,
                         extensions=extensions)

    main_loop.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = ArgumentParser("An example of training an MLP on"
                            " the MNIST dataset.")
    parser.add_argument("--num-epochs", type=int, default=1000,
                        help="Number of training epochs to do.")
    args = parser.parse_args()
    main(num_epochs=args.num_epochs)
