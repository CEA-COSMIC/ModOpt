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


def val2int(val):
    """Convert to int

    This method checks if input value is an int and if not converts it.

    Parameters
    ----------
    val : int, float, str, list, tuple or np.ndarray
        Input value

    Returns
    -------
    int interger value or np.ndarray array of ints

    Raises
    ------
    ValueError
        For invalid input type

    Examples
    --------
    >>> from modopt.base.types import val2int
    >>> x = np.arange(5).astype(float)
    >>> x
    array([ 0.,  1.,  2.,  3.,  4.])
    >>> val2int(x)
    array([0, 1, 2, 3, 4])

    """

    if isinstance(val, int):
        pass
    elif isinstance(val, (float, str)):
        val = int(float(val))
    elif isinstance(val, (list, tuple)):
        val = np.array(val, dtype=float).astype(int)
    elif isinstance(val, np.ndarray):
        if np.issubdtype(val.dtype, 'int64'):
            pass
        else:
            val = val.astype(float).astype(int)
    else:
        raise ValueError('Invalid input type.')

    return val


def val2float(val):
    """Convert to float

    This method checks if input value is a float and if not converts it.

    Parameters
    ----------
    val : int, float, str, list, tuple or np.ndarray
        Input value

    Returns
    -------
    float floating point value or np.ndarray array of floats

    Examples
    --------
    >>> from modopt.base.types import val2float
    >>> x = np.arange(5)
    >>> x
    array([0, 1, 2, 3, 4])
    >>> val2float(x)
    array([ 0.,  1.,  2.,  3.,  4.])

    """

    if isinstance(val, float):
        pass
    elif isinstance(val, (int, str)):
        val = float(val)
    elif isinstance(val, (list, tuple)):
        val = np.array(val, dtype=float)
    elif isinstance(val, np.ndarray):
        if np.issubdtype(val.dtype, 'float64'):
            pass
        else:
            val = val.astype(float)
    else:
        raise ValueError('Invalid input type.')

    return val


def val2str(val):
    """Convert to string

    This method checks if input value is a string and if not converts it.

    Parameters
    ----------
    val : int, float, str, list, tuple or np.ndarray
        Input value

    Returns
    -------
    str string or np.ndarray array of strings


    Examples
    --------
    >>> from modopt.base.types import val2str
    >>> x = np.arange(5)
    >>> x
    array([0, 1, 2, 3, 4])
    >>> val2str(x)
    array(['0', '1', '2', '3', '4'],
          dtype='|S21')

    """

    if isinstance(val, str):
        pass
    elif isinstance(val, (int, float)):
        val = str(val)
    elif isinstance(val, (list, tuple)):
        val = np.array(val, dtype=str)
    elif isinstance(val, np.ndarray):
        if np.issubdtype(val.dtype, 'S21'):
            pass
        else:
            val = val.astype(str)
    else:
        raise ValueError('Invalid input type.')

    return val


def nan2val(array, val=0.0):
    """Convert NAN to val

    This converts all NANs in an array to a specified value.

    Parameters
    ----------
    array : np.ndarray, list or tuple
        Input array
    val : int or float, optional
        Value to replace NANs. Default (val=0.0)

    Returns
    -------
    np.ndarray array without NANs

    NOTES
    -----
    Output data type defined by val type.

    Examples
    --------
    >>> from modopt.base.types import nan2val
    >>> x = [1., 2., np.nan, 4.]
    >>> x
    [1.0, 2.0, nan, 4.0]
    >>> nan2val(x, 3.)
    array([ 1.,  2.,  3.,  4.])

    """

    new_array = np.copy(array)
    new_array[np.isnan(new_array)] = val

    if isinstance(val, int):
        new_array = val2int(new_array)

    return new_array
