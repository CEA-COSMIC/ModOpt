# -*- coding: utf-8 -*-

"""GRADIENT CLASSES.

This module contains classses for defining algorithm gradients.
Based on work by Yinghao Ge and Fred Ngole.

:Author: Samuel Farrens <samuel.farrens@cea.fr>

"""

import numpy as np

from modopt.base.types import check_callable, check_float, check_npndarray


class GradParent(object):
    """Gradient Parent Class.

    This class defines the basic methods that will be inherited by specific
    gradient classes

    Parameters
    ----------
    input_data : numpy.ndarray
        The observed data
    op : function
        The operator
    trans_op : function
        The transpose operator
    get_grad : function, optional
        Method for calculating the gradient (default is ``None``)
    cost: function, optional
        Method for calculating the cost (default is ``None``)
    data_type : type, optional
        Expected data type of the input data (default is ``None``)
    verbose : bool, optional
        Option for verbose output (default is ``True``)

    Examples
    --------
    >>> import numpy as np
    >>> from modopt.opt.gradient import GradParent
    >>> y = np.arange(9).reshape(3, 3).astype(float)
    >>> g = GradParent(y, lambda x: x ** 2, lambda x: x ** 3)
    >>> g.op(y)
    array([[ 0.,  1.,  4.],
           [ 9., 16., 25.],
           [36., 49., 64.]])
    >>> g.trans_op(y)
    array([[  0.,   1.,   8.],
           [ 27.,  64., 125.],
           [216., 343., 512.]])
    >>> g.trans_op_op(y)
    array([[0.00000e+00, 1.00000e+00, 6.40000e+01],
           [7.29000e+02, 4.09600e+03, 1.56250e+04],
           [4.66560e+04, 1.17649e+05, 2.62144e+05]])

    """

    def __init__(
        self,
        input_data,
        op,
        trans_op,
        get_grad=None,
        cost=None,
        data_type=None,
        verbose=True,
    ):

        self.verbose = verbose
        self._grad_data_type = data_type
        self.obs_data = input_data
        self.op = op
        self.trans_op = trans_op

        if not isinstance(get_grad, type(None)):
            self.get_grad = get_grad
        if not isinstance(cost, type(None)):
            self.cost = cost

    @property
    def obs_data(self):
        """Observed Data."""
        return self._obs_data

    @obs_data.setter
    def obs_data(self, input_data):

        if self._grad_data_type in {float, np.floating}:
            input_data = check_float(input_data)
        check_npndarray(
            input_data,
            dtype=self._grad_data_type,
            writeable=False,
            verbose=self.verbose,
        )

        self._obs_data = input_data

    @property
    def op(self):
        """Operator."""
        return self._op

    @op.setter
    def op(self, operator):

        self._op = check_callable(operator)

    @property
    def trans_op(self):
        """Transpose operator."""
        return self._trans_op

    @trans_op.setter
    def trans_op(self, operator):

        self._trans_op = check_callable(operator)

    @property
    def get_grad(self):
        """Get gradient value."""
        return self._get_grad

    @get_grad.setter
    def get_grad(self, method):

        self._get_grad = check_callable(method)

    @property
    def grad(self):
        """Gradient value."""
        return self._grad

    @grad.setter
    def grad(self, input_value):

        if self._grad_data_type in {float, np.floating}:
            input_value = check_float(input_value)
        self._grad = input_value

    @property
    def cost(self):
        """Cost contribution."""
        return self._cost

    @cost.setter
    def cost(self, method):

        self._cost = check_callable(method)

    def trans_op_op(self, input_data):
        r"""Transpose Operation of the Operator.

        This method calculates the action of the transpose operator on
        the action of the operator on the data

        Parameters
        ----------
        input_data : numpy.ndarray
            Input data array

        Returns
        -------
        numpy.ndarray
            Result

        Notes
        -----
        Implements the following equation:

        .. math::
            \mathbf{H}^T(\mathbf{H}\mathbf{x})

        """
        return self.trans_op(self.op(input_data))


class GradBasic(GradParent):
    """Basic Gradient Class.

    This class defines the gradient calculation and costs methods for
    common inverse problems.

    Parameters
    ----------
    args : interable
        Positional arguments
    kwargs : dict
        Keyword arguments

    Examples
    --------
    >>> import numpy as np
    >>> from modopt.opt.gradient import GradBasic
    >>> y = np.arange(9).reshape(3, 3).astype(float)
    >>> g = GradBasic(y, lambda x: x ** 2, lambda x: x ** 3)
    >>> g.get_grad(y)
    >>> g.grad
    array([[0.00000e+00, 0.00000e+00, 8.00000e+00],
           [2.16000e+02, 1.72800e+03, 8.00000e+03],
           [2.70000e+04, 7.40880e+04, 1.75616e+05]])

    See Also
    --------
    GradParent : parent class

    """

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.get_grad = self._get_grad_method
        self.cost = self._cost_method

    def _get_grad_method(self, input_data):
        r"""Get the gradient.

        This method calculates the gradient step from the input data

        Parameters
        ----------
        input_data : numpy.ndarray
            Input data array

        Notes
        -----
        Implements the following equation:

        .. math::
            \nabla F(x) = \mathbf{H}^T(\mathbf{H}\mathbf{x} - \mathbf{y})

        """
        self.grad = self.trans_op(self.op(input_data) - self.obs_data)

    def _cost_method(self, *args, **kwargs):
        """Calculate gradient component of the cost.

        This method returns the l2 norm error of the difference between the
        original data and the data obtained after optimisation

        Parameters
        ----------
        args : interable
            Positional arguments
        kwargs : dict
            Keyword arguments

        Returns
        -------
        float
            Gradient cost component

        """
        cost_val = 0.5 * np.linalg.norm(self.obs_data - self.op(args[0])) ** 2

        if 'verbose' in kwargs and kwargs['verbose']:
            print(' - DATA FIDELITY (X):', cost_val)

        return cost_val
