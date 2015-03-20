#!/usr/bin/env python

import logging
from argparse import ArgumentParser

import numpy
import theano
from theano import tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams
from blocks.algorithms import (
    GradientDescent, StepClipping, CompositeRule, Adam
)
from blocks.model import Model
from blocks.extensions import FinishAfter, Timing, Printing
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.main_loop import MainLoop
from blocks.extensions import SimpleExtension
from blocks.bricks import Linear, Tanh
from blocks.bricks.lookup import LookupTable
from blocks.bricks.recurrent import SimpleRecurrent
from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate
from blocks.graph import ComputationGraph
from blocks.initialization import Uniform, Constant
from fuel.streams import DataStream
from fuel.transformers import Padding

from dl_tutorials.blocks.extensions.plot import (
    PlotManager, Plotter
)
from dl_tutorials.utils.sequences import sequence_map
from dl_tutorials.fuel.datasets.text_file import TextFile
from dl_tutorials.fuel.schemes import BatchwiseShuffledScheme
from dl_tutorials.utils.softmax import softmax


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


class PrintSamples(SimpleExtension):
    """Prints samples from a model."""
    def __init__(self, sampler, voc, **kwargs):
        kwargs.setdefault("before_first_epoch", True)
        kwargs.setdefault("on_resumption", True)
        kwargs.setdefault("after_training", True)
        kwargs.setdefault("after_epoch", True)
        kwargs.setdefault("on_interrupt", True)
        self.sampler = sampler
        self.voc = voc
        super(PrintSamples, self).__init__(**kwargs)

    def do(self, which_callback, *args):
        samples = self.sampler()
        for sample in samples:
            print(''.join([self.voc[idx] for idx in sample]))


def main(num_epochs=100):
    x = tensor.matrix('features')
    m = tensor.matrix('features_mask')

    x_int = x.astype(dtype='int32').T
    train_dataset = TextFile('inspirational.txt')
    train_dataset.indexables[0] = numpy.array(sorted(
        train_dataset.indexables[0], key=len
    ))

    n_voc = len(train_dataset.dict.keys())

    init_probs = numpy.array(
        [sum(filter(lambda idx:idx == w,
                    [s[0] for s in train_dataset.indexables[
                        train_dataset.sources.index('features')]]
                    )) for w in xrange(n_voc)],
        dtype=theano.config.floatX
    )
    init_probs = init_probs / init_probs.sum()

    n_h = 100
    linear_embedding = LookupTable(
        length=n_voc,
        dim=n_h,
        weights_init=Uniform(std=0.01),
        biases_init=Constant(0.)
    )
    linear_embedding.initialize()
    lstm_biases = numpy.zeros(4 * n_h).astype(dtype=theano.config.floatX)
    lstm_biases[n_h:(2 * n_h)] = 4.
    rnn = SimpleRecurrent(
        dim=n_h,
        activation=Tanh(),
        weights_init=Uniform(std=0.01),
        biases_init=Constant(0.)
    )
    rnn.initialize()
    score_layer = Linear(
        input_dim=n_h,
        output_dim=n_voc,
        weights_init=Uniform(std=0.01),
        biases_init=Constant(0.)
    )
    score_layer.initialize()

    embedding = (linear_embedding.apply(x_int[:-1])
                 * tensor.shape_padright(m.T[1:]))
    rnn_out = rnn.apply(inputs=embedding, mask=m.T[1:])
    probs = softmax(
        sequence_map(score_layer.apply, rnn_out, mask=m.T[1:])[0]
    )
    idx_mask = m.T[1:].nonzero()
    cost = CategoricalCrossEntropy().apply(
        x_int[1:][idx_mask[0], idx_mask[1]],
        probs[idx_mask[0], idx_mask[1]]
    )
    cost.name = 'cost'
    misclassification = MisclassificationRate().apply(
        x_int[1:][idx_mask[0], idx_mask[1]],
        probs[idx_mask[0], idx_mask[1]]
    )
    misclassification.name = 'misclassification'

    cg = ComputationGraph([cost])
    params = cg.parameters

    algorithm = GradientDescent(
        cost=cost,
        params=params,
        step_rule=Adam()
    )

    train_data_stream = Padding(
        data_stream=DataStream(
            dataset=train_dataset,
            iteration_scheme=BatchwiseShuffledScheme(
                examples=train_dataset.num_examples,
                batch_size=10,
            )
        ),
        mask_sources=('features',)
    )

    model = Model(cost)

    extensions = []
    extensions.append(Timing())
    extensions.append(FinishAfter(after_n_epochs=num_epochs))
    extensions.append(TrainingDataMonitoring(
        [cost, misclassification],
        prefix='train',
        after_epoch=True))

    batch_size = 10
    length = 30
    trng = MRG_RandomStreams(18032015)
    u = trng.uniform(size=(length, batch_size, n_voc))
    gumbel_noise = -tensor.log(-tensor.log(u))
    init_samples = (tensor.log(init_probs).dimshuffle(('x', 0))
                    + gumbel_noise[0]).argmax(axis=-1)
    init_states = rnn.initial_state('states', batch_size)

    def sampling_step(g_noise, states, samples_step):
        embedding_step = linear_embedding.apply(samples_step)
        next_states = rnn.apply(inputs=embedding_step,
                                            states=states,
                                            iterate=False)
        probs_step = softmax(score_layer.apply(next_states))
        next_samples = (tensor.log(probs_step)
                        + g_noise).argmax(axis=-1)

        return next_states, next_samples

    [_, samples], _ = theano.scan(
        fn=sampling_step,
        sequences=[gumbel_noise[1:]],
        outputs_info=[init_states, init_samples]
    )

    sampling = theano.function([], samples.owner.inputs[0].T)

    plotters = []
    plotters.append(Plotter(
        channels=[['train_cost', 'train_misclassification']],
        titles=['Costs']))

    extensions.append(PlotManager('Language modelling example',
                                  plotters=plotters,
                                  after_epoch=True,
                                  after_training=True))
    extensions.append(Printing())
    extensions.append(PrintSamples(sampler=sampling,
                                   voc=train_dataset.inv_dict))

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
