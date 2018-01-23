# -*- coding: utf-8 -*-

"""NUMPY ADJUSTMENT ROUTINES

This module contains methods for adjusting the default output for certain
Numpy functions.

:Author: Samuel Farrens <samuel.farrens@cea.fr>

"""

from __future__ import division
import numpy as np


def rotate(data):
    """Rotate

    This method rotates an input numpy array by 180 degrees.

    Parameters
    ----------
    data : np.ndarray
        Input data array (at least 2D)

    Returns
    -------
    np.ndarray rotated data

    Notes
    -----
    Adjustment to numpy.rot90()

    Examples
    --------
    >>> from modopt.base.np_adjust import rotate
    >>> x = np.arange(9).reshape((3, 3))
    >>> x
    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])
    >>> rotate(x)
    array([[8, 7, 6],
           [5, 4, 3],
           [2, 1, 0]])

    """

    return np.rot90(data, 2)


def rotate_stack(data):
    """Rotate stack

    This method rotates each array in a stack of arrays by 180 degrees.

    Parameters
    ----------
    data : np.ndarray
        Input data array (at least 3D)

    Returns
    -------
    np.ndarray rotated data

    Examples
    --------
    >>> from modopt.base.np_adjust import rotate_stack
    >>> x = np.arange(18).reshape((2, 3, 3))
    >>> x
    array([[[ 0,  1,  2],
            [ 3,  4,  5],
            [ 6,  7,  8]],
           [[ 9, 10, 11],
            [12, 13, 14],
            [15, 16, 17
    >>> rotate_stack(x)
    array([[[ 8,  7,  6],
            [ 5,  4,  3],
            [ 2,  1,  0]],
           [[17, 16, 15],
            [14, 13, 12],
            [11, 10,  9]]])

    """

    return np.array([rotate(x) for x in data])


def pad2d(data, padding):
    """Pad array

    This method pads an input numpy array with zeros in all directions.

    Parameters
    ----------
    data : np.ndarray
        Input data array (at least 2D)
    padding : int, tuple
        Amount of padding in x and y directions, respectively

    Returns
    -------
    np.ndarray padded data

    Notes
    -----
    Adjustment to numpy.pad()

    Examples
    --------
    >>> from modopt.base.np_adjust import pad2d
    >>> x = np.arange(9).reshape((3, 3))
    >>> x
    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])
    >>> pad2d(x, (1, 1))
    array([[0, 0, 0, 0, 0],
           [0, 0, 1, 2, 0],
           [0, 3, 4, 5, 0],
           [0, 6, 7, 8, 0],
           [0, 0, 0, 0, 0]])

    """

    data = np.array(data)

    if isinstance(padding, int):
        padding = np.array([padding])
    elif isinstance(padding, (tuple, list)):
        padding = np.array(padding)
    elif isinstance(padding, np.ndarray):
        pass
    else:
        raise ValueError('Padding must be an integer or a tuple (or list, '
                         'np.ndarray) of itegers')

    if padding.size == 1:
        padding = np.repeat(padding, 2)

    return np.pad(data, ((padding[0], padding[0]), (padding[1], padding[1])),
                  'constant')


def ftr(data):
    """Fancy transpose right

    Apply fancy_transpose() to data with roll=1

    Parameters
    ----------
    data : np.ndarray
        Input data array

    Returns
    -------
    np.ndarray transposed data

    """

    return fancy_transpose(data)


def ftl(data):
    """Fancy transpose left

    Apply fancy_transpose() to data with roll=-1

    Parameters
    ----------
    data : np.ndarray
        Input data array

    Returns
    -------
    np.ndarray transposed data

    """

    return fancy_transpose(data, -1)


def fancy_transpose(data, roll=1):
    """Fancy transpose

    This method transposes a multidimensional matrix.

    Parameters
    ----------
    data : np.ndarray
        Input data array
    roll : int
        Roll direction and amount. Default (roll=1)

    Returns
    -------
    np.ndarray transposed data

    Notes
    -----
    Adjustment to numpy.transpose

    Examples
    --------
    >>> from modopt.base.np_adjust import fancy_transpose
    >>> x = np.arange(27).reshape(3, 3, 3)
    >>> x
    array([[[ 0,  1,  2],
            [ 3,  4,  5],
            [ 6,  7,  8]],
           [[ 9, 10, 11],
            [12, 13, 14],
            [15, 16, 17]],
           [[18, 19, 20],
            [21, 22, 23],
            [24, 25, 26]]])
    >>> fancy_transpose(x)
    array([[[ 0,  3,  6],
            [ 9, 12, 15],
            [18, 21, 24]],
           [[ 1,  4,  7],
            [10, 13, 16],
            [19, 22, 25]],
           [[ 2,  5,  8],
            [11, 14, 17],
            [20, 23, 26]]])
    >>> fancy_transpose(x, roll=-1)
    array([[[ 0,  9, 18],
            [ 1, 10, 19],
            [ 2, 11, 20]],
           [[ 3, 12, 21],
            [ 4, 13, 22],
            [ 5, 14, 23]],
           [[ 6, 15, 24],
            [ 7, 16, 25],
            [ 8, 17, 26]]])

    """

    axis_roll = np.roll(np.arange(data.ndim), roll)

    return np.transpose(data, axes=axis_roll)
