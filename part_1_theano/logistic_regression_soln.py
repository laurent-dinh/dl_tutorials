from theano import tensor
from theano.tensor.nnet import sigmoid
from blocks.utils import shared_floatx_zeros


class LogisticRegressor(object):
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.params = [shared_floatx_zeros((input_dim, 1)),
                       shared_floatx_zeros((1,))]
        self.W = self.params[0]
        self.b = self.params[1]

    def get_probs(self, features):
        """Output the probability of being a positive class

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
        return sigmoid(features.dot(self.W) + self.b)

    def get_params(self):
        """Returns the list of parameters of the model.

        Returns
        -------
        params : list
            The list of shared variables that are parameters of the model.
        """
        return self.params

    def get_cost(self, probs, targets):
        """Output the probability of being a positive class

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
