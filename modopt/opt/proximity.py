# -*- coding: utf-8 -*-

"""PROXIMITY OPERATORS

This module contains classes of proximity operators for optimisation

:Author: Samuel Farrens <samuel.farrens@cea.fr>

:Version: 1.4

:Date: 14/12/2017

"""

from __future__ import print_function
from builtins import range
import numpy as np
from modopt.signal.noise import thresh
from modopt.signal.svd import svd_thresh, svd_thresh_coef
from modopt.opt.algorithms import ForwardBackward
from modopt.signal.positivity import positive
from modopt.math.matrix import nuclear_norm
from modopt.base.transform import *


class IdentityProx(object):
    """Identity operator class

    This is a dummy class that can be used as a proximity operator

    """

    def __init__(self):
        pass

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

    def cost(self, *args, **kwargs):
        """Calculate identity component of the cost

        This method returns 0 as the posivituty does not contribute to the
        cost.

        Returns
        -------
        float zero

        """

        return 0.0


class Positive(object):
    """Positivity proximity operator

    This class defines the positivity proximity operator

    """

    def __init__(self):
        pass

    def op(self, data, **kwargs):
        """Operator

        This method preserves only the positive coefficients of the input data

        Parameters
        ----------
        data : np.ndarray
            Input data array

        Returns
        -------
        np.ndarray all positive elements from input data

        """

        return positive(data)

    def cost(self, *args, **kwargs):
        """Calculate positivity component of the cost

        This method returns 0 as the posivituty does not contribute to the
        cost.

        Returns
        -------
        float zero

        """

        if 'verbose' in kwargs and kwargs['verbose']:
            print(' - Min (X):', np.min(args[0]))

        return 0.0


class SparseThreshold(object):
    """Threshold proximity operator

    This class defines the threshold proximity operator

    Parameters
    ----------
    linear : class
        Linear operator class
    weights : np.ndarray
        Input array of weights
    thresh_type : str {'hard', 'soft'}, optional
        Threshold type (default is 'soft')

    """

    def __init__(self, linear, weights, thresh_type='soft'):

        self._linear = linear
        self._weights = weights
        self._thresh_type = thresh_type

    def op(self, data, extra_factor=1.0):
        """Operator

        This method returns the input data thresholded by the weights

        Parameters
        ----------
        data : np.ndarray
            Input data array
        extra_factor : float
            Additional multiplication factor

        Returns
        -------
        np.ndarray thresholded data

        """

        threshold = self._weights * extra_factor

        return thresh(data, threshold, self._thresh_type)

    def cost(self, *args, **kwargs):
        """Calculate sparsity component of the cost

        This method returns the l1 norm error of the weighted wavelet
        coefficients

        Returns
        -------
        float sparsity cost component

        """

        cost_val = np.sum(np.abs(self._weights * self._linear.op(args[0])))

        if 'verbose' in kwargs and kwargs['verbose']:
            print(' - L1 NORM (X):', cost_val)

        return cost_val


class LowRankMatrix(object):
    """Low-rank proximity operator

    This class defines the low-rank proximity operator

    Parameters
    ----------
    thresh : float
        Threshold value
    treshold_type : str {'hard', 'soft'}
        Threshold type (options are 'hard' or 'soft')
    lowr_type : str {'standard', 'ngole'}
        Low-rank implementation (options are 'standard' or 'ngole')
    operator : class
        Operator class ('ngole' only)

    """

    def __init__(self, thresh, thresh_type='soft',
                 lowr_type='standard', operator=None):

        self.thresh = thresh
        self.thresh_type = thresh_type
        self.lowr_type = lowr_type
        self.operator = operator

    def op(self, data, extra_factor=1.0):
        """Operator

        This method returns the input data after the singular values have been
        thresholded

        Parameters
        ----------
        data : np.ndarray
            Input data array
        extra_factor : float
            Additional multiplication factor

        Returns
        -------
        np.ndarray SVD thresholded data

        """

        # Update threshold with extra factor.
        threshold = self.thresh * extra_factor

        if self.lowr_type == 'standard':
            data_matrix = svd_thresh(cube2matrix(data), threshold,
                                     thresh_type=self.thresh_type)

        elif self.lowr_type == 'ngole':
            data_matrix = svd_thresh_coef(cube2matrix(data), self.operator,
                                          threshold,
                                          thresh_type=self.thresh_type)

        new_data = matrix2cube(data_matrix, data.shape[1:])

        # Return updated data.
        return new_data

    def cost(self, *args, **kwargs):
        """Calculate low-rank component of the cost

        This method returns the nuclear norm error of the deconvolved data in
        matrix form

        Returns
        -------
        float low-rank cost component

        """

        cost_val = self.thresh * nuclear_norm(cube2matrix(args[0]))

        if 'verbose' in kwargs and kwargs['verbose']:
            print(' - NUCLEAR NORM (X):', cost_val)

        return cost_val


class ProximityCombo(object):
    """Proximity Combo

    This class defines a combined proximity operator

    Parameters
    ----------
    operators : list
        List of proximity operator class instances

    """

    def __init__(self, operators):

        self.operators = operators

    def op(self, data, extra_factor=1.0):
        """Operator

        This method returns the result of applying all of the proximity
        operators to the data

        Parameters
        ----------
        data : np.ndarray
            Input data array
        extra_factor : float
            Additional multiplication factor

        Returns
        -------
        np.ndarray result

        """

        res = np.empty(len(self.operators), dtype=np.ndarray)

        for i in range(len(self.operators)):
            res[i] = self.operators[i].op(data[i], extra_factor=extra_factor)

        return res

    def cost(self, *args, **kwargs):
        """Calculate combined proximity operator components of the cost

        This method returns the sum of the cost components from each of the
        proximity operators

        Returns
        -------
        float combinded cost components

        """

        return np.sum([op.cost(*args, **kwargs) for op in self.operators])


class SubIter(object):
    """Sub iteration operator

    This class defines the sub-iteration proximity operator

    Parameters
    ----------
    data_shape : tuple
        Shape of input data array
    operator : class
        Proximity operator class
    weights : np.ndarray
        Array of weights
    u_init : np.ndarray
        Initial estimate of u

    """

    def __init__(self, data_shape, operator, weights=None, u_init=None):

        self.operator = operator

        if not isinstance(weights, type(None)):
            self.weights = weights

        if isinstance(u_init, type(None)):
            self.u = np.ones(data_shape)

        self.opt = ForwardBackward(self.u, self.operator,
                                   Threshold(self.weights), auto_iterate=False,
                                   indent_level=2)

    def update_weights(self, weights):
        """Update weights

        This method updates the values of the weights

        Parameters
        ----------
        weights : np.ndarray
            Array of weights

        """

        self.weights = weights

    def update_u(self):
        """Update u

        This method updates the values of u

        """

        self.opt.iterate(100)
        self.u = self.opt.x_final

    def op(self, data):
        """Operator

        Parameters
        ----------
        data : np.ndarray
            Input data array

        Returns
        -------
        np.ndarray result

        """

        self.update_u()

        new_data = data - self.operator.adj_op(self.u)

        return new_data
