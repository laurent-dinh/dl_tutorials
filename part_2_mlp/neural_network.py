import numpy
import theano
from theano import tensor
from theano.tensor.nnet import sigmoid
from blocks.utils import shared_floatx_zeros
floatX = theano.config.floatX


class NeuralNetwork(object):
    def __init__(self, input_dim, n_hidden):
        # WRITEME
        pass

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
        # WRITEME
        pass

    def get_params(self):
        """Returns the list of parameters of the class.

        Returns
        -------
        params : list
            The list of shared variables that are parameters of the class.
        """
        # WRITEME
        pass

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
        # WRITEME
        pass

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

        # WRITEME
        pass
