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
        self.W = shared_floatx_zeros((num_filters, num_channels)
                                     + filter_size)
        self.W.set_value(
            .1 * (numpy.random.uniform(
                size=self.W.get_value().shape
                ).astype(floatX) - 0.5
            )
        )
        self.b = shared_floatx_zeros((num_filters,))
        self.params = [self.W, self.b]
        self.num_filters = num_filters
        self.num_channels = num_channels
        self.filter_size = filter_size
        self.border_mode = border_mode
        if border_mode not in ['full', 'valid']:
            raise ValueError('Invalid mode: must be `valid` or `full`.')
        self.step = step
        self.set_input_shape(input_shape)

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
        rval = conv2d(
            features, self.W,
            image_shape=(None, self.num_channels) + self.input_shape,
            subsample=self.step,
            border_mode=self.border_mode,
            filter_shape=self.W.get_value().shape
        )
        rval += self.b.dimshuffle(('x', 0, 'x', 'x'))

        return rval * (rval > 0.)

    def get_params(self):
        """Returns the list of parameters of the model.

        Returns
        -------
        params : list
            The list of shared variables that are parameters of the model.
        """
        return self.params

    def get_weights(self):
        """Returns the weights parameter of the model.

        Returns
        -------
        weights : :class:`~tensor.sharedvar.SharedVariable`
            The weights of the model connected to the input.
        """
        return self.W

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
        if input_shape is (None, None) or input_shape is None:
            input_shape = self.input_shape
        axis_0, axis_1 = input_shape
        if axis_0 is None or axis_1 is None:
            return input_shape
        if self.border_mode == 'full':
            output_axis_0 = (axis_0 + self.filter_size[0]) \
                / self.step[0] - 1
            output_axis_1 = (axis_1 + self.filter_size[1]) \
                / self.step[1] - 1
        else:
            output_axis_0 = (axis_0 - self.filter_size[0]) \
                / self.step[0] + 1
            output_axis_1 = (axis_1 - self.filter_size[1]) \
                / self.step[1] + 1
        output_shape = (output_axis_0, output_axis_1)

        return output_shape

    def set_input_shape(self, input_shape):
        """Get the output shape given an input shape.

        Parameters
        ----------
        input_shape : tuple
            The shape tuple must have length 2 and
            represent the axes (0, 1).

        """
        self.input_shape = input_shape


class MaxPooling(object):
    def __init__(self, pooling_size=(2, 2),
                 input_shape=(None, None)):
        self.pooling_size = pooling_size
        self.input_shape = input_shape

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
        rval = max_pool_2d(features, self.pooling_size, ignore_border=True)
        return rval

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
        if input_shape is (None, None) or input_shape is None:
            input_shape = self.input_shape
        axis_0, axis_1 = input_shape
        if axis_0 is None or axis_1 is None:
            return input_shape
        output_axis_0 = axis_0 / self.pooling_size[0]
        output_axis_1 = axis_1 / self.pooling_size[1]
        output_shape = (output_axis_0, output_axis_1)

        return output_shape

    def set_input_shape(self, input_shape):
        """Get the output shape given an input shape.

        Parameters
        ----------
        input_shape : tuple
            The shape tuple must have length 2 and
            represent the axes (0, 1).

        """
        self.input_shape = input_shape


class ConvolutionalNetwork(object):
    def __init__(self, layers, input_shape=(None, None)):
        self.layers = layers
        self.set_input_shape(input_shape)

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
        rval = features
        for layer in self.layers:
            rval = layer.apply(rval)
        return rval

    def get_params(self):
        """Returns the list of parameters of the model.

        Returns
        -------
        params : list
            The list of shared variables that are parameters of the model.
        """
        params = []
        for layer in self.layers:
            params += layer.get_params()
        return params

    def get_weights(self):
        """Returns the weights parameter of the model.

        Returns
        -------
        weights : :class:`~tensor.sharedvar.SharedVariable`
            The weights of the model connected to the input.
        """
        return self.layers[0].get_weights()

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
        if input_shape is (None, None) or input_shape is None:
            input_shape = self.input_shape
        axis_0, axis_1 = input_shape
        if axis_0 is None or axis_1 is None:
            return input_shape
        output_shape = input_shape
        for layer in self.layers:
            output_shape = layer.get_output_shape(output_shape)

        return output_shape

    def set_input_shape(self, input_shape):
        """Get the output shape given an input shape.

        Parameters
        ----------
        input_shape : tuple
            The shape tuple must have length 2 and
            represent the axes (0, 1).

        """
        self.input_shape = input_shape
        output_shape = input_shape
        for layer in self.layers:
            layer.set_input_shape(output_shape)
            output_shape = layer.get_output_shape(output_shape)
