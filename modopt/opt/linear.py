# -*- coding: utf-8 -*-

"""LINEAR OPERATORS

This module contains linear operator classes.

:Author: Samuel Farrens <samuel.farrens@cea.fr>

:Version: 1.3

:Date: 19/07/2017

"""

from builtins import range, zip
import numpy as np
from modopt.base.wrappers import add_agrs_kwargs
from modopt.math.matrix import rotate
from modopt.signal.wavelet import *


class LinearParent(object):
    r"""Linear Operator Parent Class

    This class sets the structure for defining linear operator instances.

    Parameters
    ----------
    op : func
        Callable function that implements the linear operation
    adj_op : func
        Callable function that implements the linear adjoint operation

    Examples
    --------
    >>> from modopt.opt.linear import LinearParent
    >>> a = LinearParent(lambda x: x * 2, lambda x: x ** 3)
    >>> a.op(2)
    4
    >>> a.adj_op(2)
    8

    """

    def __init__(self, op, adj_op):

        self.op = op
        self.adj_op = adj_op

    def _test_operator(self, operator):
        """ Test Input Operator

        This method checks if the input operator is a callable funciton and
        adds support for `*args` and `**kwargs` if not already provided

        Parameters
        ----------
        operator : func
            Callable function

        Returns
        -------
        func wrapped by `add_agrs_kwargs`

        Raises
        ------
        TypeError
            For invalid input type

        """

        if not callable(operator):
            raise TypeError('The input operator must be a callable function.')

        return add_agrs_kwargs(operator)

    @property
    def op(self):
        """Linear Operator

        This method defines the linear operator

        """

        return self._op

    @op.setter
    def op(self, operator):

        self._op = self._test_operator(operator)

    @property
    def adj_op(self):
        """Linear Adjoint Operator

        This method defines the linear operator

        """

        return self._adj_op

    @adj_op.setter
    def adj_op(self, operator):

        self._adj_op = self._test_operator(operator)


class Identity(LinearParent):
    """Identity operator class

    This is a dummy class that can be used in the optimisation classes

    """

    def __init__(self):

        self.l1norm = 1.0
        self.op = lambda x: x
        self.adj_op = self.op


class Wavelet(LinearParent):
    """Wavelet class

    This class defines the wavelet transform operators

    Parameters
    ----------
    data : np.ndarray
        Input data array, normally an array of 2D images
    wavelet_opt: str, optional
        Additional options for `mr_transform`

    """

    def __init__(self, data, wavelet_opt=''):

        self.y = data
        self.data_shape = data.shape[-2:]
        n = data.shape[0]

        self.filters = get_mr_filters(self.data_shape, opt=wavelet_opt)
        self.l1norm = n * np.sqrt(sum((np.sum(np.abs(filter)) ** 2 for
                                       filter in self.filters)))
        self.op = lambda x: filter_convolve_stack(x, self.filters)
        self.adj_op = lambda x: filter_convolve_stack(x, self.filters,
                                                      filter_rot=True)


class LinearCombo(object):
    """Linear combination class

    This class defines a combination of linear transform operators

    Parameters
    ----------
    operators : list
        List of linear operator class instances
    weights : list, optional
        List of weights for combining the linear adjoint operator results

    """

    def __init__(self, operators, weights=None):

        self.operators = operators
        self.weights = weights
        self.l1norm = np.array([operator.l1norm for operator in
                                self.operators])

    def op(self, data):
        """Operator

        This method returns the input data operated on by all of the operators

        Parameters
        ----------
        data : np.ndarray
            Input data array, a 2D image

        Returns
        -------
        np.ndarray linear operation results

        """

        res = np.empty(len(self.operators), dtype=np.ndarray)

        for i in range(len(self.operators)):
            res[i] = self.operators[i].op(data)

        return res

    def adj_op(self, data):
        """Adjoint operator

        This method returns the combination of the result of all of the
        adjoint operators. If weights are provided the comibination is the sum
        of the weighted results, otherwise the combination is the mean.

        Parameters
        ----------
        data : np.ndarray
            Input data array, an array of coefficients

        Returns
        -------
        np.ndarray adjoint operation results

        """

        if isinstance(self.weights, type(None)):

            return np.mean([operator.adj_op(x) for x, operator in
                           zip(data, self.operators)], axis=0)

        else:

            return np.sum([weight * operator.adj_op(x) for x, operator,
                          weight in zip(data, self.operators, weights)],
                          axis=0)
