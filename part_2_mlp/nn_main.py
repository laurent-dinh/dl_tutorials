#!/usr/bin/env python

import logging
from argparse import ArgumentParser
import copy

import theano
from theano import tensor
from blocks.algorithms import GradientDescent, Momentum
from blocks.model import Model
from blocks.extensions import FinishAfter, Timing, Printing
from blocks.extensions.monitoring import (DataStreamMonitoring,
                                          TrainingDataMonitoring)
from blocks.main_loop import MainLoop
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
from fuel.transformers import Mapping, ForceFloatX

from dl_tutorials.blocks.extensions.plot import (
    PlotManager, Plotter, Display2DData
)
from dl_tutorials.part_2_mlp.neural_network import NeuralNetwork
from dl_tutorials.utils.build_2d_datasets import build_2d_datasets


# Getting around having tuples as argument and output
class TupleMapping(object):
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, args):
        return (self.fn(args[0]), )


def main(dataset_name='sklearn', num_epochs=100):
    x = tensor.matrix('features')
    y = tensor.lmatrix('targets')

    neural_net = NeuralNetwork(input_dim=2, n_hidden=[20])
    probs = neural_net.get_probs(features=x)
    params = neural_net.get_params()
    cost = neural_net.get_cost(probs=probs, targets=y).mean()
    cost.name = 'cost'
    misclassification = neural_net.get_misclassification(
        probs=probs, targets=y
    ).mean()
    misclassification.name = 'misclassification'

    train_dataset, test_dataset = build_2d_datasets(dataset_name=dataset_name)

    algorithm = GradientDescent(
        cost=cost,
        params=params,
        step_rule=Momentum(learning_rate=0.1,
                           momentum=0.1))

    train_data_stream = DataStream(
        dataset=train_dataset,
        iteration_scheme=SequentialScheme(
            examples=train_dataset.num_examples,
            batch_size=40,
        )
    )
    valid_data_stream = ForceFloatX(
        data_stream=DataStream(
            dataset=test_dataset,
            iteration_scheme=SequentialScheme(
                examples=range(10) + range(200, 210),
                batch_size=400,
            )
        )
    )
    test_data_stream = ForceFloatX(
        data_stream=DataStream(
            dataset=test_dataset,
            iteration_scheme=SequentialScheme(
                examples=test_dataset.num_examples,
                batch_size=400,
            )
        )
    )

    model = Model(cost)

    extensions = []
    extensions.append(Timing())
    extensions.append(FinishAfter(after_n_epochs=num_epochs))
    extensions.append(DataStreamMonitoring(
        [cost, misclassification],
        test_data_stream,
        prefix='test'))
    extensions.append(DataStreamMonitoring(
        [cost, misclassification],
        valid_data_stream,
        prefix='valid'))
    extensions.append(TrainingDataMonitoring(
        [cost, misclassification],
        prefix='train',
        after_epoch=True))

    scoring_function = theano.function([x], probs)

    plotters = []
    plotters.append(Plotter(
        channels=[['test_cost', 'test_misclassification',
                   'train_cost', 'train_misclassification']],
        titles=['Costs']))
    score_train_stream = Mapping(data_stream=copy.deepcopy(train_data_stream),
                                 mapping=TupleMapping(scoring_function),
                                 add_sources=('scores',))
    score_test_stream = Mapping(data_stream=copy.deepcopy(test_data_stream),
                                mapping=TupleMapping(scoring_function),
                                add_sources=('scores',))
    plotters.append(Display2DData(
        data_streams=[score_train_stream, copy.deepcopy(train_data_stream),
                      score_test_stream, copy.deepcopy(test_data_stream)],
        radius=0.01
    ))

    extensions.append(PlotManager('2D examples', plotters=plotters,
                                  after_epoch=False,
                                  every_n_epochs=50,
                                  after_training=True))
    extensions.append(Printing())
    main_loop = MainLoop(model=model,
                         data_stream=train_data_stream,
                         algorithm=algorithm,
                         extensions=extensions)

    main_loop.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = ArgumentParser("An example of training an MLP on"
                            " a 2D dataset.")
    parser.add_argument("--num-epochs", type=int, default=100,
                        help="Number of training epochs to do.")
    parser.add_argument("--dataset", default="sklearn", nargs="?",
                        help=("Dataset to use."))
    args = parser.parse_args()
    main(dataset_name=args.dataset, num_epochs=args.num_epochs)
