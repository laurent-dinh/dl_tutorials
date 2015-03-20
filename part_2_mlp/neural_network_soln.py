import numpy
import theano
from theano import tensor
from theano.tensor.nnet import sigmoid
from blocks.utils import shared_floatx_zeros
floatX = theano.config.floatX


class NeuralNetwork(object):
    def __init__(self, input_dim, n_hidden):
        dim_list = [input_dim] + n_hidden

        self.W = []
        self.b = []

        for i in xrange(len(dim_list) - 1):
            in_dim = dim_list[i]
            out_dim = dim_list[i + 1]
            W = shared_floatx_zeros((in_dim, out_dim))
            W.set_value(
                .1 * (numpy.random.uniform(
                    size=W.get_value().shape
                    ).astype(floatX) - 0.5
                )
            )
            W.name = 'W_' + str(i)
            b = shared_floatx_zeros((out_dim, ))
            b.name = 'b_' + str(i)
            self.W.append(W)
            self.b.append(b)

        W = shared_floatx_zeros((n_hidden[-1], 1))
        W.set_value(
            .1 * (numpy.random.uniform(
                size=W.get_value().shape
                ).astype(floatX) - 0.5
            )
        )
        b = shared_floatx_zeros((1,))
        W.name = 'W_out'
        b.name = 'b_out'
        self.W.append(W)
        self.b.append(b)

    def get_probs(self, features):
        """Output the probability of being a positive.

        Parameters
        ----------
        features : :class:`~tensor.TensorVariable`
            The features that you consider as input.
            Must have shape (batch_size, input_dim).

        Returns
        -------
        probs : :class:`~tensor.TensorVariable`
            The probabilities of each example to belong to
            the positive class. Must have shape (batch_size, 1)
        """
        out = features
        for W, b in zip(self.W, self.b):
            out = out.dot(W) + b
            if W != self.W[-1]:
                out = (out > 0.) * (out)
        return sigmoid(out)

    def get_params(self):
        """Returns the list of parameters of the class.

        Returns
        -------
        params : list
            The list of shared variables that are parameters of the class.
        """
        return self.W + self.b

    def get_cost(self, probs, targets):
        """Output the logistic loss.

        Parameters
        ----------
        probs : :class:`~tensor.TensorVariable`
            The probabilities of each example to belong to
            the positive class. Must have shape (batch_size, 1)
        targets : :class:`~tensor.TensorVariable`
            The indicator on whether the example belongs to the
            positive class. Must have shape (batch_size, 1)

        Returns
        -------
        cost : :class:`~tensor.TensorVariable`
            The corresponding logistic cost.
            .. math:: -targets \log(probs) - (1 - targets) \log(1 - probs)
        """
        return - targets * tensor.log(probs) \
            - (1 - targets) * tensor.log(1 - probs)

    def get_misclassification(self, probs, targets):
        """Output the misclassification error.

        This misclassification is done when classifying an example as
        the most likely class.

        Parameters
        ----------
        probs : :class:`~tensor.TensorVariable`
            The probabilities of each example to belong to
            the positive class. Must have shape (batch_size, 1)
        targets : :class:`~tensor.TensorVariable`
            The indicator on whether the example belongs to the
            positive class. Must have shape (batch_size, 1)

        Returns
        -------
        misclassification : :class:`~tensor.TensorVariable`
            The corresponding misclassification error, if we classify
            an example as the most likely class.
        """

        return targets * (probs < 0.5) \
            + (1 - targets) * (probs > 0.5)
