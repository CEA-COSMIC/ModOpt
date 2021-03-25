# -*- coding: utf-8 -*-

"""TYPE HANDLING ROUTINES.

This module contains methods for handing object types.

:Author: Samuel Farrens <samuel.farrens@cea.fr>

"""

import numpy as np

from modopt.base.wrappers import add_args_kwargs
from modopt.interface.errors import warn


def check_callable(input_obj, add_agrs=True):
    """Check input object is callable.

    This method checks if the input operator is a callable funciton and
    optionally adds support for arguments and keyword arguments if not already
    provided

    Parameters
    ----------
    input_obj : callable
        Callable function
    add_agrs : bool, optional
        Option to add support for agrs and kwargs (default is ``True``)

    Returns
    -------
    function
        Function wrapped by `add_args_kwargs`

    Raises
    ------
    TypeError
        For invalid input type

    See Also
    --------
    modopt.base.wrappers.add_args_kwargs : wrapper used

    """
    if not callable(input_obj):
        raise TypeError('The input object must be a callable function.')

    if add_agrs:
        input_obj = add_args_kwargs(input_obj)

    return input_obj


def check_float(input_obj):
    """Check Float.

    Check if input object is a float or a numpy.ndarray of floats, if not
    convert.

    Parameters
    ----------
    input_obj : any
        Input value

    Returns
    -------
    float or numpy.ndarray
        Input value as a float

    Raises
    ------
    TypeError
        For invalid input type

    Examples
    --------
    >>> import numpy as np
    >>> from modopt.base.types import check_float
    >>> a = np.arange(5)
    >>> a
    array([0, 1, 2, 3, 4])
    >>> check_float(a)
    array([0., 1., 2., 3., 4.])

    See Also
    --------
    check_int : related function

    """
    if not isinstance(input_obj, (int, float, list, tuple, np.ndarray)):
        raise TypeError('Invalid input type.')
    if isinstance(input_obj, int):
        input_obj = float(input_obj)
    elif isinstance(input_obj, (list, tuple)):
        input_obj = np.array(input_obj, dtype=float)
    elif (
        isinstance(input_obj, np.ndarray)
        and (not np.issubdtype(input_obj.dtype, np.floating))
    ):
        input_obj = input_obj.astype(float)

    return input_obj


def check_int(input_obj):
    """Check Integer.

    Check if input value is an int or a np.ndarray of ints, if not convert.

    Parameters
    ----------
    input_obj : any
        Input value

    Returns
    -------
    int or numpy.ndarray
        Input value as an integer

    Raises
    ------
    TypeError
        For invalid input type

    Examples
    --------
    >>> import numpy as np
    >>> from modopt.base.types import check_int
    >>> a = np.arange(5).astype(float)
    >>> a
    array([0., 1., 2., 3., 4.])
    >>> check_int(a)
    array([0, 1, 2, 3, 4])

    See Also
    --------
    check_float : related function

    """
    if not isinstance(input_obj, (int, float, list, tuple, np.ndarray)):
        raise TypeError('Invalid input type.')
    if isinstance(input_obj, float):
        input_obj = int(input_obj)
    elif isinstance(input_obj, (list, tuple)):
        input_obj = np.array(input_obj, dtype=int)
    elif (
        isinstance(input_obj, np.ndarray)
        and (not np.issubdtype(input_obj.dtype, np.integer))
    ):
        input_obj = input_obj.astype(int)

    return input_obj


def check_npndarray(input_obj, dtype=None, writeable=True, verbose=True):
    """Check Numpy ND-Array.

    Check if input object is a numpy array.

    Parameters
    ----------
    input_obj : numpy.ndarray
        Input object
    dtype : type
        Numpy ndarray data type
    writeable : bool
        Option to make array immutable
    verbose : bool
        Verbosity option

    Raises
    ------
    TypeError
        For invalid input type
    TypeError
        For invalid numpy.ndarray dtype

    """
    if not isinstance(input_obj, np.ndarray):
        raise TypeError('Input is not a numpy array.')

    if (
        (not isinstance(dtype, type(None)))
        and (not np.issubdtype(input_obj.dtype, dtype))
    ):
        raise (
            TypeError(
                'The numpy array elements are not of type: {0}'.format(dtype),
            ),
        )

    if not writeable and verbose and input_obj.flags.writeable:
        warn('Making input data immutable.')

    input_obj.flags.writeable = writeable
