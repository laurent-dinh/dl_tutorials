from theano import tensor
from blocks.utils import shared_floatx_zeros
from dl_tutorials.utils.softmax import softmax


class SoftmaxRegressor(object):
    def __init__(self, input_dim, n_classes):
        # WRITEME
        pass

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
        # WRITEME
        pass

    def get_params(self):
        """Returns the list of parameters of the model.

        Returns
        -------
        params : list
            The list of shared variables that are parameters of the model.
        """
        # WRITEME
        pass

    def get_weights(self):
        """Returns the weights parameter of the model.

        Returns
        -------
        weights : :class:`~tensor.sharedvar.SharedVariable`
            The weights of the model connected to the input.
        """
        # WRITEME
        pass

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

        # WRITEME
        pass
