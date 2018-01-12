"""VALIDATION ROUTINES

This module contains methods for testing signal and operator properties.

:Author: Samuel Farrens <samuel.farrens@cea.fr>

"""

from __future__ import print_function
import numpy as np


def transpose_test(operator, operator_t, x_shape, x_args=None, y_shape=None,
                   y_args=None):
    r"""Transpose test

    This method tests two operators to see if they are the transpose of each
    other.

    Parameters
    ----------
    operator : function
        Operator function
    operator_t : function
        Transpose operator function
    x_shape : tuple
        Shape of operator input data
    x_args : tuple
        Arguments to be passed to operator
    y_shape : tuple, optional
        Shape of transpose operator input data
    y_args : tuple, optional
        Arguments to be passed to transpose operator

    Examples
    --------
    >>> from modopt.signal.validation import transpose_test
    >>> np.random.seed(2)
    >>> transpose_test(lambda x, y: x.dot(y), lambda x, y: x.dot(y.T),
    (3, 3), x_args=x)
     - |<MX, Y> - <X, M.TY>| = 0.0

    """

    if not callable(operator) or not callable(operator_t):
        raise TypeError('The input operators must be callable functions.')

    if isinstance(y_shape, type(None)):
        y_shape = x_shape

    if isinstance(y_args, type(None)):
        y_args = x_args

    # Generate random arrays.
    x = np.random.ranf(x_shape)
    y = np.random.ranf(y_shape)

    # Calculate <MX, Y>
    mx_y = np.sum(np.multiply(operator(x, x_args), y))

    # Calculate <X, M.TY>
    x_mty = np.sum(np.multiply(x, operator_t(y, y_args)))

    # Test the difference between the two.
    print(' - |<MX, Y> - <X, M.TY>| =', np.abs(mx_y - x_mty))
