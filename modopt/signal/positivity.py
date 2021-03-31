# -*- coding: utf-8 -*-

"""POSITIVITY.

This module contains a function that retains only positive coefficients in
an array

:Author: Samuel Farrens <samuel.farrens@cea.fr>

"""

import numpy as np


def pos_thresh(input_data):
    """Positive Threshold.

    Keep only positive coefficients from input data.

    Parameters
    ----------
    input_data : int, float, list, tuple or numpy.ndarray
        Input data

    Returns
    -------
    int, float, or numpy.ndarray
        Positive coefficients

    """
    return input_data * (input_data > 0)


def pos_recursive(input_data):
    """Positive Recursive.

    Run pos_thresh on input array or recursively for ragged nested arrays.

    Parameters
    ----------
    input_data : numpy.ndarray
        Input data

    Returns
    -------
    numpy.ndarray
        Positive coefficients

    """
    if input_data.dtype == 'O':
        res = np.array([pos_recursive(elem) for elem in input_data])

    else:
        res = pos_thresh(input_data)

    return res


def positive(input_data, ragged=False):
    """Positivity operator.

    This method preserves only the positive coefficients of the input data, all
    negative coefficients are set to zero

    Parameters
    ----------
    input_data : int, float, or numpy.ndarray
        Input data
    ragged : bool, optional
        Specify if the input_data is a ragged nested array
        (defaul is ``False``)

    Returns
    -------
    int or float, or numpy.ndarray
        Positive coefficients

    Raises
    ------
    TypeError
        For invalid input type.

    Examples
    --------
    >>> import numpy as np
    >>> from modopt.signal.positivity import positive
    >>> a = np.arange(9).reshape(3, 3) - 5
    >>> a
    array([[-5, -4, -3],
           [-2, -1,  0],
           [ 1,  2,  3]])
    >>> positive(a)
    array([[0, 0, 0],
           [0, 0, 0],
           [1, 2, 3]])

    """
    if not isinstance(input_data, (int, float, list, tuple, np.ndarray)):
        raise TypeError(
            'Invalid data type, input must be `int`, `float`, `list`, '
            + '`tuple` or `np.ndarray`.',
        )

    if isinstance(input_data, (int, float)):
        return pos_thresh(input_data)

    if ragged:
        input_data = np.array(input_data, dtype='object')

    else:
        input_data = np.array(input_data)

    return pos_recursive(input_data)
