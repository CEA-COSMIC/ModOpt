# -*- coding: utf-8 -*-

"""TYPE HANDLING ROUTINES

This module contains methods for handing object types.

:Author: Samuel Farrens <samuel.farrens@cea.fr>

"""

import numpy as np
from modopt.base.wrappers import add_args_kwargs
from modopt.interface.errors import warn


def check_callable(val, add_agrs=True):
    r""" Check input object is callable

    This method checks if the input operator is a callable funciton and
    optionally adds support for arguments and keyword arguments if not already
    provided

    Parameters
    ----------
    val : function
        Callable function
    add_agrs : bool, optional
        Option to add support for agrs and kwargs

    Returns
    -------
    func wrapped by `add_args_kwargs`

    Raises
    ------
    TypeError
        For invalid input type

    """

    if not callable(val):
        raise TypeError('The input object must be a callable function.')

    if add_agrs:
        val = add_args_kwargs(val)

    return val


def check_float(val):
    r"""Check if input value is a float or a np.ndarray of floats, if not
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

    if not isinstance(val, (int, float, list, tuple, np.ndarray)):
        raise TypeError('Invalid input type.')
    if isinstance(val, int):
        val = float(val)
    elif isinstance(val, (list, tuple)):
        val = np.array(val, dtype=float)
    elif isinstance(val, np.ndarray) and (not np.issubdtype(val.dtype,
                                                            np.floating)):
        val = val.astype(float)

    return val


def check_int(val):
    r"""Check if input value is an int or a np.ndarray of ints, if not convert.

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

    if not isinstance(val, (int, float, list, tuple, np.ndarray)):
        raise TypeError('Invalid input type.')
    if isinstance(val, float):
        val = int(val)
    elif isinstance(val, (list, tuple)):
        val = np.array(val, dtype=int)
    elif isinstance(val, np.ndarray) and (not np.issubdtype(val.dtype,
                                                            np.integer)):
        val = val.astype(int)

    return val


def check_npndarray(val, dtype=None, writeable=True, verbose=True):
    """Check if input object is a numpy array.

    Parameters
    ----------
    val : np.ndarray
        Input object

    """

    if not isinstance(val, np.ndarray):
        raise TypeError('Input is not a numpy array.')

    if ((not isinstance(dtype, type(None))) and
            (not np.issubdtype(val.dtype, dtype))):
        raise TypeError('The numpy array elements are not of type: {}'
                        ''.format(dtype))

    if not writeable and verbose and val.flags.writeable:
        warn('Making input data immutable.')

    val.flags.writeable = writeable
