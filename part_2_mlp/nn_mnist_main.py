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
from dl_tutorials.part_2_mlp.neural_softmax import NeuralSoftmax


# Getting around having tuples as argument and output
class TupleMapping(object):
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, args):
        return (self.fn(args[0]), )


def main(num_epochs=10000):
    x = tensor.matrix('features')
    y = tensor.lmatrix('targets')

    neural_net = NeuralSoftmax(input_dim=784, n_classes=10, n_hidden=[100])
    probs = neural_net.get_probs(features=x)
    params = neural_net.get_params()
    weights = neural_net.get_weights()
    cost = neural_net.get_cost(probs=probs, targets=y).mean()
    cost.name = 'cost'
    misclassification = neural_net.get_misclassification(
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
                examples=range(50000),
                batch_size=100,
            )
        )
    )
    valid_data_stream = ForceFloatX(
        data_stream=DataStream(
            dataset=train_dataset,
            iteration_scheme=SequentialScheme(
                examples=range(50000, 60000),
                batch_size=1000,
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
    extensions.append(DataStreamMonitoring(
        [cost, misclassification],
        valid_data_stream,
        prefix='valid'))
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
        rescale=2.
    )
    images_displayer = DisplayImage(
        image_getters=[display_train, weight_display],
        titles=['Training examples', 'Softmax weights']
    )
    plotters.append(images_displayer)

    extensions.append(PlotManager('MNIST neural network examples',
                                  plotters=plotters,
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
    parser = ArgumentParser("An example of training an MLP on"
                            " the MNIST dataset.")
    parser.add_argument("--num-epochs", type=int, default=10000,
                        help="Number of training epochs to do.")
    args = parser.parse_args()
    main(num_epochs=args.num_epochs)
