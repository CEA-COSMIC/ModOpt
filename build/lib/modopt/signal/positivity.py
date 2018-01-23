# -*- coding: utf-8 -*-

"""POSITIVITY

This module contains a function that retains only positive coefficients in
an array

:Author: Samuel Farrens <samuel.farrens@cea.fr>

"""

import numpy as np


def positive(data):
    r"""Positivity operator

    This method preserves only the positive coefficients of the input data, all
    negative coefficients are set to zero

    Parameters
    ----------
    data : int, float, list, tuple or np.ndarray
        Input data

    Returns
    -------
    int or float, or np.ndarray array with only positive coefficients

    Raises
    ------
    TypeError
        For invalid input type.

    Examples
    --------
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

    if not isinstance(data, (int, float, list, tuple, np.ndarray)):
        raise TypeError('Invalid data type, input must be `int`, `float`, '
                        '`list`, `tuple` or `np.ndarray`.')

    def pos_thresh(data):

        return data * (data > 0)

    def pos_recursive(data):

        data = np.array(data)

        if not data.dtype == 'O':

            result = list(pos_thresh(data))

        else:

            result = [pos_recursive(x) for x in data]

        return result

    if isinstance(data, (int, float)):

        return pos_thresh(data)

    else:

        return np.array(pos_recursive(data))
