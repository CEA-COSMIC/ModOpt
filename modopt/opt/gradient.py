# -*- coding: utf-8 -*-

"""GRADIENT CLASSES

This module contains classses for defining algorithm gradients.
Based on work by Yinghao Ge and Fred Ngole.

:Author: Samuel Farrens <samuel.farrens@cea.fr>

"""

import numpy as np
from modopt.base.types import check_callable, check_float, check_npndarray


class GradParent(object):
    r"""Gradient Parent Class

    This class defines the basic methods that will be inherited by specific
    gradient classes

    Parameters
    ----------
    data : np.ndarray
        The observed data
    op : function
        The operator
    trans_op : function
        The transpose operator
    get_grad : function, optional
        Method for calculating the gradient (default is None)
    cost: function, optional
        Method for calculating the cost (default is None)
    data_type : type, optional
        Expected data type of the input data (default is None)

    Examples
    --------
    >>> from modopt.opt.gradient import GradParent
    >>> y = np.arange(9).reshape(3, 3).astype(float)
    >>> g = GradParent(y, lambda x: x ** 2, lambda x: x ** 3)
    >>> g.op(y)
    array([[  0.,   1.,   4.],
           [  9.,  16.,  25.],
           [ 36.,  49.,  64.]])
    >>> g.trans_op(y)
    array([[   0.,    1.,    8.],
           [  27.,   64.,  125.],
           [ 216.,  343.,  512.]])
    >>> g.trans_op_op(y)
    array([[  0.00000000e+00,   1.00000000e+00,   6.40000000e+01],
           [  7.29000000e+02,   4.09600000e+03,   1.56250000e+04],
           [  4.66560000e+04,   1.17649000e+05,   2.62144000e+05]])

    """

    def __init__(self, data, op, trans_op, get_grad=None, cost=None,
                 data_type=None):

        self._grad_data_type = data_type
        self.obs_data = data
        self.op = op
        self.trans_op = trans_op

        if not isinstance(get_grad, type(None)):
            self.get_grad = get_grad
        if not isinstance(cost, type(None)):
            self.cost = cost

    @property
    def obs_data(self):
        """Observed Data

        Raises
        ------
        TypeError
            For invalid input type

        """

        return self._obs_data

    @obs_data.setter
    def obs_data(self, data):

        if self._grad_data_type in (float, np.floating):
            data = check_float(data)
        check_npndarray(data, dtype=self._grad_data_type, writeable=False)

        self._obs_data = data

    @property
    def op(self):
        """Operator

        This method defines the operator

        """

        return self._op

    @op.setter
    def op(self, operator):

        self._op = check_callable(operator)

    @property
    def trans_op(self):
        """Transpose Operator

        This method defines the transpose operator

        """

        return self._trans_op

    @trans_op.setter
    def trans_op(self, operator):

        self._trans_op = check_callable(operator)

    @property
    def get_grad(self):
        """Get Gradient

        This method defines the calculation of the gradient

        """

        return self._get_grad

    @get_grad.setter
    def get_grad(self, method):

        self._get_grad = check_callable(method)

    @property
    def grad(self):
        """Gradient

        The gradient value

        """

        return self._grad

    @grad.setter
    def grad(self, value):

        if self._grad_data_type in (float, np.floating):
            value = check_float(value)
        self._grad = value

    @property
    def cost(self):
        """Cost Contribution

        This method defines the proximity operator's contribution to the total
        cost

        """

        return self._cost

    @cost.setter
    def cost(self, method):

        self._cost = check_callable(method)

    def trans_op_op(self, data):
        r"""Transpose Operation of the Operator

        This method calculates the action of the transpose operator on
        the action of the operator on the data

        Parameters
        ----------
        data : np.ndarray
            Input data array

        Returns
        -------
        np.ndarray result

        Notes
        -----
        Implements the following equation:

        .. math::
            \mathbf{H}^T(\mathbf{H}\mathbf{x})

        """

        return self.trans_op(self.op(data))


class GradBasic(GradParent):
    r"""Basic Gradient Class

    This class defines the gradient calculation and costs methods for
    common inverse problems

    Examples
    --------
    >>> from modopt.opt.gradient import GradBasic
    >>> y = np.arange(9).reshape(3, 3).astype(float)
    >>> g = GradBasic(y, lambda x: x ** 2, lambda x: x ** 3)
    >>> g.get_grad(y)
    >>> g.grad
    array([[  0.00000000e+00,   0.00000000e+00,   8.00000000e+00],
           [  2.16000000e+02,   1.72800000e+03,   8.00000000e+03],
           [  2.70000000e+04,   7.40880000e+04,   1.75616000e+05]])

    """

    def __init__(self, *args, **kwargs):

        super(GradBasic, self).__init__(*args, **kwargs)
        self.get_grad = self._get_grad_method
        self.cost = self._cost_method

    def _get_grad_method(self, data):
        r"""Get the gradient

        This method calculates the gradient step from the input data

        Parameters
        ----------
        data : np.ndarray
            Input data array

        Notes
        -----
        Implements the following equation:

        .. math::
            \nabla F(x) = \mathbf{H}^T(\mathbf{H}\mathbf{x} - \mathbf{y})

        """

        self.grad = self.trans_op(self.op(data) - self.obs_data)

    def _cost_method(self, *args, **kwargs):
        """Calculate gradient component of the cost

        This method returns the l2 norm error of the difference between the
        original data and the data obtained after optimisation

        Returns
        -------
        float gradient cost component

        """

        cost_val = 0.5 * np.linalg.norm(self.obs_data - self.op(args[0])) ** 2

        if 'verbose' in kwargs and kwargs['verbose']:
            print(' - DATA FIDELITY (X):', cost_val)

        return cost_val
