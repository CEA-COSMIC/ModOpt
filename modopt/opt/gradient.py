# -*- coding: utf-8 -*-

"""GRADIENT CLASSES

This module contains classses for defining algorithm gradients.
Based on work by Yinghao Ge and Fred Ngole.

:Author: Samuel Farrens <samuel.farrens@cea.fr>

:Version: 1.3

:Date: 19/07/2017

"""

import numpy as np
from modopt.base.types import check_callable


class GradParent(object):
    r"""Gradient Parent Class

    This class defines the basic methods that will be inherited by specific
    gradient classes

    Parameters
    ----------
    y : np.ndarray
        The observed data
    op : function
        The operator
    op_trans : function
        The transpose operator

    Examples
    --------
    >>> from modopt.opt.gradient import *
    >>> y = np.arange(9).reshape(3, 3).astype(float)
    >>> g = GradParent(y, lambda x: x ** 2, lambda x: x ** 3)
    >>> g.MX(y)
    array([[  0.,   1.,   4.],
           [  9.,  16.,  25.],
           [ 36.,  49.,  64.]])
    >>> g.MtX(y)
    array([[   0.,    1.,    8.],
           [  27.,   64.,  125.],
           [ 216.,  343.,  512.]])
    >>> g.MtMX(y)
    array([[  0.00000000e+00,   1.00000000e+00,   6.40000000e+01],
           [  7.29000000e+02,   4.09600000e+03,   1.56250000e+04],
           [  4.66560000e+04,   1.17649000e+05,   2.62144000e+05]])
    >>> g.get_grad(y)
    >>> g.grad
    array([[  0.00000000e+00,   0.00000000e+00,   8.00000000e+00],
           [  2.16000000e+02,   1.72800000e+03,   8.00000000e+03],
           [  2.70000000e+04,   7.40880000e+04,   1.75616000e+05]])

    """

    def __init__(self, y, op, op_trans):

        self.y = y
        self.MX = op
        self.MtX = op_trans

    @property
    def y(self):
        """Observed Data

        Raises
        ------
        TypeError
            For invalid input type

        """

        return self._y

    @y.setter
    def y(self, data):

        if ((not isinstance(data, np.ndarray)) or
                (not np.issubdtype(data.dtype, float))):

            raise TypeError('Invalid input type, input data must be a '
                            'numpy array of floats.')

        self._y = data

    @property
    def MX(self):
        """Operator

        This method defines the operator

        """

        return self._MX

    @MX.setter
    def MX(self, operator):

        self._MX = check_callable(operator)

    @property
    def MtX(self):
        """Operator

        This method defines the transpose operator

        """

        return self._MtX

    @MtX.setter
    def MtX(self, operator):

        self._MtX = check_callable(operator)

    def MtMX(self, x):
        """M^T M X

        This method calculates the action of the transpose of the matrix M on
        the action of the matrix M on the data X

        Parameters
        ----------
        x : np.ndarray
            Input data array

        Returns
        -------
        np.ndarray result

        Notes
        -----
        Calculates  M^T (MX)

        """

        return self.MtX(self.MX(x))

    def get_grad(self, x):
        """Get the gradient step

        This method calculates the gradient step from the input data

        Parameters
        ----------
        x : np.ndarray
            Input data array

        Returns
        -------
        np.ndarray gradient value

        Notes
        -----

        Calculates M^T (MX - Y)

        """

        self.grad = self.MtX(self.MX(x) - self.y)

    def cost(self, *args, **kwargs):
        """Calculate gradient component of the cost

        This method returns the l2 norm error of the difference between the
        original data and the data obtained after optimisation

        Returns
        -------
        float gradient cost component

        """

        cost_val = 0.5 * np.linalg.norm(self.y - self.MX(args[0])) ** 2

        if 'verbose' in kwargs and kwargs['verbose']:
            print(' - DATA FIDELITY (X):', cost_val)

        return cost_val
