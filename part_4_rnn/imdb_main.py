#!/usr/bin/env python

import logging
from argparse import ArgumentParser

import numpy
import theano
from theano import tensor
from blocks.algorithms import (
    GradientDescent, StepClipping, CompositeRule, Adam
)
from blocks.model import Model
from blocks.extensions import FinishAfter, Timing, Printing
from blocks.extensions.monitoring import (DataStreamMonitoring,
                                          TrainingDataMonitoring)
from blocks.main_loop import MainLoop
from blocks.bricks import Linear, Sigmoid
from blocks.bricks.lookup import LookupTable
from blocks.bricks.recurrent import LSTM
from blocks.graph import ComputationGraph
from blocks.initialization import Uniform, Constant
from fuel.streams import DataStream
from fuel.transformers import Padding

from dl_tutorials.blocks.extensions.plot import (
    PlotManager, Plotter
)
from dl_tutorials.fuel.datasets.imdb import IMDB
from dl_tutorials.fuel.schemes import BatchwiseShuffledScheme


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


def main(num_epochs=100):
    x = tensor.matrix('features')
    m = tensor.matrix('features_mask')
    y = tensor.imatrix('targets')

    x_int = x.astype(dtype='int32').T
    train_dataset = IMDB()
    idx_sort = numpy.argsort(
        [len(s) for s in
         train_dataset.indexables[
             train_dataset.sources.index('features')]]
    )
    n_voc = len(train_dataset.dict.keys())
    for idx in xrange(len(train_dataset.sources)):
        train_dataset.indexables[idx] = train_dataset.indexables[idx][idx_sort]

    n_h = 100
    linear_embedding = LookupTable(
        length=n_voc,
        dim=4 * n_h,
        weights_init=Uniform(std=0.01),
        biases_init=Constant(0.)
    )
    linear_embedding.initialize()
    lstm_biases = numpy.zeros(4 * n_h).astype(dtype=theano.config.floatX)
    lstm_biases[n_h:(2 * n_h)] = 4.
    rnn = LSTM(
        dim=n_h,
        weights_init=Uniform(std=0.01),
        biases_init=Constant(0.)
    )
    rnn.initialize()
    score_layer = Linear(
        input_dim=n_h,
        output_dim=1,
        weights_init=Uniform(std=0.01),
        biases_init=Constant(0.)
    )
    score_layer.initialize()

    embedding = linear_embedding.apply(x_int) * tensor.shape_padright(m.T)
    rnn_out = rnn.apply(embedding)
    rnn_out_mean_pooled = rnn_out[0][-1]

    probs = Sigmoid().apply(
        score_layer.apply(rnn_out_mean_pooled))

    cost = - (y * tensor.log(probs)
              + (1 - y) * tensor.log(1 - probs)
              ).mean()
    cost.name = 'cost'

    misclassification = (y * (probs < 0.5)
                         + (1 - y) * (probs > 0.5)
                         ).mean()
    misclassification.name = 'misclassification'

    cg = ComputationGraph([cost])
    params = cg.parameters

    algorithm = GradientDescent(
        cost=cost,
        params=params,
        step_rule=CompositeRule(
            components=[StepClipping(threshold=10.),
                        Adam()
                        ]
        )
    )

    n_train = int(numpy.floor(.8 * train_dataset.num_examples))
    n_valid = int(numpy.floor(.1 * train_dataset.num_examples))
    train_data_stream = Padding(
        data_stream=DataStream(
            dataset=train_dataset,
            iteration_scheme=BatchwiseShuffledScheme(
                examples=range(n_train),
                batch_size=10,
            )
        ),
        mask_sources=('features',)
    )
    valid_data_stream = Padding(
        data_stream=DataStream(
            dataset=train_dataset,
            iteration_scheme=BatchwiseShuffledScheme(
                examples=range(n_train, n_train + n_valid),
                batch_size=10,
            )
        ),
        mask_sources=('features',)
    )
    test_data_stream = Padding(
        data_stream=DataStream(
            dataset=train_dataset,
            iteration_scheme=BatchwiseShuffledScheme(
                examples=range(n_train + n_valid,
                               train_dataset.num_examples),
                batch_size=10,
            )
        ),
        mask_sources=('features',)
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
        channels=[['train_cost', 'train_misclassification',
                   'valid_cost', 'valid_misclassification']],
        titles=['Costs']))

    extensions.append(PlotManager('IMDB classification example',
                                  plotters=plotters,
                                  after_epoch=True,
                                  after_training=True))
    extensions.append(Printing())

    main_loop = MainLoop(model=model,
                         data_stream=train_data_stream,
                         algorithm=algorithm,
                         extensions=extensions)

    main_loop.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = ArgumentParser("An example of training a recurrent net on"
                            " a text dataset.")
    parser.add_argument("--num-epochs", type=int, default=100,
                        help="Number of training epochs to do.")
    args = parser.parse_args()
    main(num_epochs=args.num_epochs)
