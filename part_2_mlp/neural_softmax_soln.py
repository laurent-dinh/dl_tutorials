import numpy
import theano
from theano import tensor
from dl_tutorials.utils.softmax import softmax
from blocks.utils import shared_floatx_zeros
floatX = theano.config.floatX


class NeuralSoftmax(object):
    def __init__(self, input_dim, n_hidden, n_classes):
        print floatX
        dim_list = [input_dim] + n_hidden

        self.W = []
        self.b = []

        for i in xrange(len(dim_list) - 1):
            in_dim = dim_list[i]
            out_dim = dim_list[i + 1]
            W = shared_floatx_zeros((in_dim, out_dim))
            W.set_value(
                .01 * (numpy.random.uniform(
                    size=W.get_value().shape
                    ).astype(floatX) - 0.5
                )
            )
            W.name = 'W_' + str(i)
            b = shared_floatx_zeros((out_dim, ))
            b.name = 'b_' + str(i)
            self.W.append(W)
            self.b.append(b)

        W = shared_floatx_zeros((n_hidden[-1], n_classes))
        W.set_value(
            .01 * (numpy.random.uniform(
                size=W.get_value().shape
                ).astype(floatX) - 0.5
            )
        )
        b = shared_floatx_zeros((n_classes,))
        W.name = 'W_out'
        b.name = 'b_out'
        self.W.append(W)
        self.b.append(b)

    def get_probs(self, features):
        """Output the probability of belonging to a class

        Parameters
        ----------
        features : :class:`~tensor.TensorVariable`
            The features that you consider as input.
            Must have shape (batch_size, input_dim).

        Returns
        -------
        probs : :class:`~tensor.TensorVariable`
            The probabilities of each example of belonging to
            each class. Must have shape (batch_size, n_classes)
        """
        out = features
        for W, b in zip(self.W, self.b):
            out = out.dot(W) + b
            if W != self.W[-1]:
                out = (out > 0.) * (out)
        return softmax(out)

    def get_params(self):
        """Returns the list of parameters of the model.

        Returns
        -------
        params : list
            The list of shared variables that are parameters of the model.
        """
        return self.W + self.b

    def get_weights(self):
        """Returns the weights parameter of the model.

        Returns
        -------
        weights : :class:`~tensor.sharedvar.SharedVariable`
            The weights of the model connected to the input.
        """
        return self.W[0]

    def get_cost(self, probs, targets):
        """Output the softmax loss.

        Parameters
        ----------
        probs : :class:`~tensor.TensorVariable`
            The probabilities of each example to belong to
            each class. Must have shape (batch_size, n_classes)
        targets : :class:`~tensor.TensorVariable`
            The indicator of the example class.
            Must have shape (batch_size, 1)

        Returns
        -------
        cost : :class:`~tensor.TensorVariable`
            The corresponding logistic cost.
            .. math:: - \log(probs_{targets})
        """
        return - tensor.log(
            probs[tensor.arange(probs.shape[0]),
                  targets.flatten()]
        )

    def get_misclassification(self, probs, targets):
        """Output the misclassification error.

        This misclassification is done when classifying an example as
        the most likely class.

        Parameters
        ----------
        probs : :class:`~tensor.TensorVariable`
            The probabilities of each example to belong to
            each class. Must have shape (batch_size, n_classes)
        targets : :class:`~tensor.TensorVariable`
            The indicator of the example class.
            Must have shape (batch_size, 1)

        Returns
        -------
        misclassification : :class:`~tensor.TensorVariable`
            The corresponding misclassification error, if we classify
            an example as the most likely class.
        """
        return tensor.neq(probs.argmax(axis=1), targets.flatten())
