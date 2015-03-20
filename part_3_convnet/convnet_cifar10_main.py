#!/usr/bin/env python

import logging
from argparse import ArgumentParser
import copy

import numpy
import theano
from theano import tensor
from blocks.algorithms import GradientDescent, Momentum
from blocks.model import Model
from blocks.extensions import FinishAfter, Timing, Printing
from blocks.extensions.monitoring import (DataStreamMonitoring,
                                          TrainingDataMonitoring)
from blocks.main_loop import MainLoop
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme, ShuffledScheme
from fuel.transformers import Mapping, ForceFloatX
from fuel.datasets import CIFAR10

from dl_tutorials.blocks.extensions.plot import (
    PlotManager, Plotter, DisplayImage
)
from dl_tutorials.blocks.extensions.display import (
    ImageDataStreamDisplay, WeightDisplay
)
from dl_tutorials.part_2_mlp.neural_softmax import NeuralSoftmax
from dl_tutorials.part_3_convnet.convolutional_network import (
    ConvolutionalNetwork,
    MaxPooling,
    ConvolutionalLayer
)


# Getting around having tuples as argument and output
class TupleMapping(object):
    def __init__(self, fn, same_len_out=False, same_len_in=False):
        self.fn = fn
        self.same_len_out = same_len_out
        self.same_len_in = same_len_in

    def __call__(self, args):
        if self.same_len_in:
            rval = (self.fn(*args), )
        else:
            rval = (self.fn(args[0]), )
        if self.same_len_out:
            rval += args[1:]
        return rval


# Rescaling the data
class Rescale(object):
    def __init__(self, scale=1., shift=0.):
        self.scale = scale
        self.shift = shift

    def __call__(self, x):
        return (self.scale * (x[0] + self.shift),) + x[1:]


def main(num_epochs=100):
    x = tensor.tensor4('features')
    y = tensor.lmatrix('targets')

    num_filters = 64
    layers = [ConvolutionalLayer(filter_size=(8, 8), num_filters=num_filters,
                                 num_channels=3, step=(1, 1),
                                 border_mode='valid'),
              MaxPooling(pooling_size=(2, 2)),
              ConvolutionalLayer(filter_size=(5, 5), num_filters=num_filters,
                                 num_channels=num_filters, step=(1, 1),
                                 border_mode='valid')]
    convnet = ConvolutionalNetwork(layers=layers)
    output_shape = convnet.get_output_shape((32, 32))
    convnet.set_input_shape((32, 32))
    fc_net = NeuralSoftmax(input_dim=numpy.prod(output_shape) * num_filters,
                           n_classes=10, n_hidden=[500])

    convnet_features = convnet.apply(x)
    probs = fc_net.get_probs(features=convnet_features.flatten(2))
    params = convnet.get_params() + fc_net.get_params()
    weights = convnet.get_weights()
    cost = fc_net.get_cost(probs=probs, targets=y).mean()
    cost.name = 'cost'
    misclassification = fc_net.get_misclassification(
        probs=probs, targets=y
    ).mean()
    misclassification.name = 'misclassification'

    train_dataset = CIFAR10('train', flatten=False)
    test_dataset = CIFAR10('test', flatten=False)

    algorithm = GradientDescent(
        cost=cost,
        params=params,
        step_rule=Momentum(learning_rate=.01,
                           momentum=0.1))

    train_data_stream = Mapping(
        data_stream=ForceFloatX(
            data_stream=DataStream(
                dataset=train_dataset,
                iteration_scheme=ShuffledScheme(
                    examples=range(50000),
                    batch_size=100,
                )
            )
        ),
        mapping=Rescale(scale=1/127.5, shift=-127.5)
    )
    valid_data_stream = Mapping(
        data_stream=ForceFloatX(
            data_stream=DataStream(
                dataset=train_dataset,
                iteration_scheme=SequentialScheme(
                    examples=range(40000, 50000),
                    batch_size=1000,
                )
            )
        ),
        mapping=Rescale(scale=1/127.5, shift=-127.5)
    )
    test_data_stream = Mapping(
        data_stream=ForceFloatX(
            data_stream=DataStream(
                dataset=test_dataset,
                iteration_scheme=SequentialScheme(
                    examples=test_dataset.num_examples,
                    batch_size=100,
                )
            )
        ),
        mapping=Rescale(scale=1/127.5, shift=-127.5)
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
        image_shape=(3, 32, 32),
        axes=('c', 0, 1),
        shift=0,
        rescale=1.,
    )
    weight_display = WeightDisplay(
        weights=weights,
        transpose=(0, 1, 2, 3),
        image_shape=(3, 8, 8),
        axes=('c', 0, 1),
        shift=-0.5,
        rescale=2.,
    )

    # Feature maps
    one_example_train_data_stream = Mapping(
        data_stream=ForceFloatX(
            data_stream=DataStream(
                dataset=train_dataset,
                iteration_scheme=ShuffledScheme(
                    examples=train_dataset.num_examples,
                    batch_size=1,
                )
            )
        ),
        mapping=Rescale(scale=1/127.5, shift=-127.5)
    )
    displayable_convnet_features = convnet_features.dimshuffle((1, 0, 2, 3))
    convnet_features_normalizer = abs(
        displayable_convnet_features
    ).max(axis=(1, 2, 3))
    displayable_convnet_features = displayable_convnet_features \
        / convnet_features_normalizer.dimshuffle((0, 'x', 'x', 'x'))
    get_displayable_convnet_features = theano.function(
        [x], displayable_convnet_features
    )
    display_feature_maps_data_stream = Mapping(
        data_stream=one_example_train_data_stream,
        mapping=TupleMapping(get_displayable_convnet_features,
                             same_len_out=True)
    )
    display_feature_maps = ImageDataStreamDisplay(
        data_stream=display_feature_maps_data_stream,
        image_shape=(1, ) + output_shape,
        axes=('c', 0, 1),
        shift=0,
        rescale=1.,
    )

    # Saliency map
    displayable_saliency_map = tensor.grad(cost, x)
    saliency_map_normalizer = abs(
        displayable_saliency_map
    ).max(axis=(1, 2, 3))
    displayable_saliency_map = displayable_saliency_map \
        / saliency_map_normalizer.dimshuffle((0, 'x', 'x', 'x'))
    get_displayable_saliency_map = theano.function(
        [x, y], displayable_saliency_map
    )

    display_saliency_map_data_stream = Mapping(
        data_stream=copy.deepcopy(train_data_stream),
        mapping=TupleMapping(get_displayable_saliency_map,
                             same_len_out=True,
                             same_len_in=True)
    )
    display_saliency_map = ImageDataStreamDisplay(
        data_stream=display_saliency_map_data_stream,
        image_shape=(3, 32, 32),
        axes=('c', 0, 1),
        shift=0,
        rescale=1.,
    )

    # Deconvolution
    x_repeated = x.repeat(num_filters, axis=0)
    convnet_features_repeated = convnet.apply(x_repeated)
    convnet_features_selected = convnet_features_repeated \
        * tensor.eye(num_filters).repeat(
            x.shape[0], axis=0
        ).dimshuffle((0, 1, 'x', 'x'))
    displayable_deconvolution = tensor.grad(misclassification, x_repeated,
                                            known_grads={
                                                convnet_features_selected:
                                                    convnet_features_selected
                                            })
    deconvolution_normalizer = abs(
        displayable_deconvolution
    ).max(axis=(1, 2, 3))
    displayable_deconvolution = displayable_deconvolution \
        / deconvolution_normalizer.dimshuffle((0, 'x', 'x', 'x'))
    get_displayable_deconvolution = theano.function(
        [x], displayable_deconvolution
    )
    display_deconvolution_data_stream = Mapping(
        data_stream=one_example_train_data_stream,
        mapping=TupleMapping(get_displayable_deconvolution,
                             same_len_out=True)
    )
    display_deconvolution = ImageDataStreamDisplay(
        data_stream=display_deconvolution_data_stream,
        image_shape=(3, 32, 32),
        axes=('c', 0, 1),
        shift=0,
        rescale=1.,
    )

    images_displayer = DisplayImage(
        image_getters=[display_train, weight_display, display_feature_maps,
                       display_saliency_map, display_deconvolution],
        titles=['Training examples', 'Convolutional weights', 'Feature maps',
                'Sensitivity', 'Deconvolution']
    )
    plotters.append(images_displayer)

    extensions.append(PlotManager('CIFAR-10 convnet examples',
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
    parser = ArgumentParser("An example of training a convnet on"
                            " the CIFAR-10 dataset.")
    parser.add_argument("--num-epochs", type=int, default=100,
                        help="Number of training epochs to do.")
    args = parser.parse_args()
    main(num_epochs=args.num_epochs)
