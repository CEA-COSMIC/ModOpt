# -*- coding: utf-8 -*-

"""TYPE HANDLING ROUTINES

This module contains methods for handing object types.

:Author: Samuel Farrens <samuel.farrens@gmail.com>

:Version: 1.0

:Date: 04/04/2017

"""


import numpy as np


def check_float(val):
    """Check if input value is a float or a np.ndarray of floats, if not
    convert.

    Parameters
    ----------
    val : any
        Input value

    Returns
    -------
    float or np.ndarray of floats

    Examples
    --------
    >>> from modopt.base.types import check_float
    >>> a = np.arange(5)
    >>> a
    array([0, 1, 2, 3, 4])
    >>> check_float(a)
    array([ 0.,  1.,  2.,  3.,  4.])

    """

    if type(val) is float:
        pass
    elif type(val) is int:
        val = float(val)
    elif type(val) is list or type(val) is tuple:
        val = np.array(val, dtype=float)
    elif type(val) is np.ndarray and val.dtype is not 'float64':
        val = val.astype(float)
    else:
        raise ValueError('Invalid input type.')

    return val


def check_int(val):
    """Check if input value is an int or a np.ndarray of ints, if not convert.

    Parameters
    ----------
    val : any
        Input value

    Returns
    -------
    int or np.ndarray of ints

    Examples
    --------
    >>> from modopt.base.types import check_int
    >>> a = np.arange(5).astype(float)
    >>> a
    array([ 0.,  1.,  2.,  3.,  4.])
    >>> check_float(a)
    array([0, 1, 2, 3, 4])

    """

    if type(val) is int:
        pass
    elif type(val) is float:
        val = int(val)
    elif type(val) is list or type(val) is tuple:
        val = np.array(val, dtype=int)
    elif type(val) is np.ndarray and val.dtype is not 'int64':
        val = val.astype(int)
    else:
        raise ValueError('Invalid input type.')

    return val
