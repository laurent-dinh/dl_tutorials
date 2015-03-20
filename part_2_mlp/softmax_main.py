#!/usr/bin/env python

import logging
from argparse import ArgumentParser
import copy

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

from dl_tutorials.blocks.extensions.plot import (
    PlotManager, Plotter, DisplayImage
)
from dl_tutorials.blocks.extensions.display import (
    ImageDataStreamDisplay, WeightDisplay
)
from dl_tutorials.part_2_mlp.softmax_regression import SoftmaxRegressor


# Getting around having tuples as argument and output
class TupleMapping(object):
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, args):
        return (self.fn(args[0]), )


def main(num_epochs=1000):
    x = tensor.matrix('features')
    y = tensor.lmatrix('targets')

    softmax_regressor = SoftmaxRegressor(input_dim=784, n_classes=10)
    probs = softmax_regressor.get_probs(features=x)
    params = softmax_regressor.get_params()
    weights = softmax_regressor.get_weights()
    cost = softmax_regressor.get_cost(probs=probs, targets=y).mean()
    cost.name = 'cost'
    misclassification = softmax_regressor.get_misclassification(
        probs=probs, targets=y
    ).mean()
    misclassification.name = 'misclassification'

    train_dataset = MNIST('train')
    test_dataset = MNIST('test')

    algorithm = GradientDescent(
        cost=cost,
        params=params,
        step_rule=Momentum(learning_rate=0.1,
                           momentum=0.1))

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

    model = Model(cost)

    extensions = []
    extensions.append(Timing())
    extensions.append(FinishAfter(after_n_epochs=num_epochs))
    extensions.append(DataStreamMonitoring(
        [cost, misclassification],
        test_data_stream,
        prefix='test'))
    extensions.append(TrainingDataMonitoring(
        [cost, misclassification],
        prefix='train',
        after_epoch=True))

    plotters = []
    plotters.append(Plotter(
        channels=[['test_cost', 'test_misclassification',
                   'train_cost', 'train_misclassification']],
        titles=['Costs']))
    display_train = ImageDataStreamDisplay(
        data_stream=copy.deepcopy(train_data_stream),
        image_shape=(28, 28, 1),
        axes=(0, 1, 'c'),
        shift=-0.5,
        rescale=2.,
    )
    weight_display = WeightDisplay(
        weights=weights,
        transpose=(1, 0),
        image_shape=(28, 28, 1),
        axes=(0, 1, 'c'),
        shift=-0.5,
        rescale=2.,
        grid_shape=(1, 10)
    )
    images_displayer = DisplayImage(
        image_getters=[display_train, weight_display],
        titles=['Training examples', 'Softmax weights']
    )
    plotters.append(images_displayer)

    extensions.append(PlotManager('MNIST softmax examples', plotters=plotters,
                                  after_epoch=False,
                                  every_n_epochs=10,
                                  after_training=True))
    extensions.append(Printing())
    main_loop = MainLoop(model=model,
                         data_stream=train_data_stream,
                         algorithm=algorithm,
                         extensions=extensions)

    main_loop.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = ArgumentParser("An example of training a softmax regression on"
                            " the MNIST dataset.")
    parser.add_argument("--num-epochs", type=int, default=1000,
                        help="Number of training epochs to do.")
    args = parser.parse_args()
    main(num_epochs=args.num_epochs)
