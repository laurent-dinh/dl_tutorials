import numpy
import theano
from theano import tensor
from theano.tensor.nnet import conv2d
from theano.tensor.signal.downsample import max_pool_2d
from blocks.utils import shared_floatx_zeros
floatX = theano.config.floatX


class ConvolutionalLayer(object):
    def __init__(self, filter_size=(3, 3), num_filters=128,
                 num_channels=3, step=(1, 1),
                 border_mode='valid', input_shape=(None, None)):
        # WRITEME
        pass

    def apply(self, features):
        """Apply the convolution

        Parameters
        ----------
        features : :class:`~tensor.TensorVariable`
            The features that you consider as input.
            Must have shape ('b', 'c', 0, 1).

        Returns
        -------
        rval : :class:`~tensor.TensorVariable`
            The output of the convolution.
            Must have shape ('b', 'c', 0, 1).
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

    def get_output_shape(self, input_shape=(None, None)):
        """Get the output shape given an input shape.

        Parameters
        ----------
        input_shape : tuple, optional
            The shape tuple must have length 2 and
            represent the axes (0, 1). Default is
            the attribute input_shape if available.

        Returns
        -------
        output_shape : tuple
            The shape tuple must have length 2 and
            represent the axes (0, 1).
        """
        # WRITEME
        pass

    def set_input_shape(self, input_shape):
        """Get the output shape given an input shape.

        Parameters
        ----------
        input_shape : tuple
            The shape tuple must have length 2 and
            represent the axes (0, 1).

        """
        # WRITEME
        pass


class MaxPooling(object):
    def __init__(self, pooling_size=(2, 2),
                 input_shape=(None, None)):
        # WRITEME
        pass

    def apply(self, features):
        """Apply the convolution

        Parameters
        ----------
        features : :class:`~tensor.TensorVariable`
            The features that you consider as input.
            Must have shape ('b', 'c', 0, 1).

        Returns
        -------
        rval : :class:`~tensor.TensorVariable`
            The output of the max pooling.
            Must have shape ('b', 'c', 0, 1).
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
        return []

    def get_output_shape(self, input_shape=(None, None)):
        """Get the output shape given an input shape.

        Parameters
        ----------
        input_shape : tuple
            The shape tuple must have length 2 and
            represent the axes (0, 1). Default is
            the attribute input_shape if available.

        Returns
        -------
        output_shape : tuple
            The shape tuple must have length 2 and
            represent the axes (0, 1).
        """
        # WRITEME
        pass

    def set_input_shape(self, input_shape):
        """Get the output shape given an input shape.

        Parameters
        ----------
        input_shape : tuple
            The shape tuple must have length 2 and
            represent the axes (0, 1).

        """
        # WRITEME
        pass


class ConvolutionalNetwork(object):
    def __init__(self, layers, input_shape=(None, None)):
        # WRITEME
        pass

    def apply(self, features):
        """Apply the convolutions and poolings in order

        Parameters
        ----------
        features : :class:`~tensor.TensorVariable`
            The features that you consider as input.
            Must have shape ('b', 'c', 0, 1).

        Returns
        -------
        rval : :class:`~tensor.TensorVariable`
            The output of the convolutional network.
            Must have shape ('b', 'c', 0, 1).
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

    def get_output_shape(self, input_shape):
        """Get the output shape given an input shape.

        Parameters
        ----------
        input_shape : tuple
            The shape tuple must have length 2 and
            represent the axes (0, 1). Default is
            the attribute input_shape if available.

        Returns
        -------
        output_shape : tuple
            The shape tuple must have length 2 and
            represent the axes (0, 1).
        """
        # WRITEME
        pass

    def set_input_shape(self, input_shape):
        """Get the output shape given an input shape.

        Parameters
        ----------
        input_shape : tuple
            The shape tuple must have length 2 and
            represent the axes (0, 1).

        """
        # WRITEME
        pass
