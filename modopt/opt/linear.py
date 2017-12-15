# -*- coding: utf-8 -*-

"""LINEAR OPERATORS

This module contains linear operator classes.

:Author: Samuel Farrens <samuel.farrens@cea.fr>

:Version: 1.3

:Date: 19/07/2017

"""

from builtins import range, zip
import numpy as np
from modopt.signal.wavelet import *
from modopt.math.matrix import rotate


class Identity(object):
    """Identity operator class

    This is a dummy class that can be used in the optimisation classes

    """

    def __init__(self):

        self.l1norm = 1.0

    def op(self, data, **kwargs):
        """Operator

        Returns the input data unchanged

        Parameters
        ----------
        data : np.ndarray
            Input data array
        **kwargs
            Arbitrary keyword arguments

        Returns
        -------
        np.ndarray input data

        """

        return data

    def adj_op(self, data):
        """Adjoint operator

        Returns the input data unchanged

        Parameters
        ----------
        data : np.ndarray
            Input data array

        Returns
        -------
        np.ndarray input data

        """

        return data


class Wavelet(object):
    """Wavelet class

    This class defines the wavelet transform operators

    Parameters
    ----------
    data : np.ndarray
        Input data array, normally an array of 2D images
    wavelet_opt: str, optional
        Additional options for `mr_transform`

    """

    def __init__(self, data, wavelet_opt=None):

        self.y = data
        self.data_shape = data.shape[-2:]
        n = data.shape[0]

        self.filters = get_mr_filters(self.data_shape, opt=wavelet_opt)
        self.l1norm = n * np.sqrt(sum((np.sum(np.abs(filter)) ** 2 for
                                       filter in self.filters)))

    def op(self, data):
        """Operator

        This method returns the input data convolved with the wavelet filters

        Parameters
        ----------
        data : np.ndarray
            Input data array, a 2D image

        Returns
        -------
        np.ndarray wavelet convolved data

        """

        return filter_convolve_stack(data, self.filters)

    def adj_op(self, data):
        """Adjoint operator

        This method returns the input data convolved with the wavelet filters
        rotated by 180 degrees

        Parameters
        ----------
        data : np.ndarray
            Input data array, a 3D of wavelet coefficients

        Returns
        -------
        np.ndarray wavelet convolved data

        """

        return filter_convolve_stack(data, self.filters, filter_rot=True)


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
