# -*- coding: utf-8 -*-

"""NUMPY ADJUSTMENT ROUTINES.

This module contains methods for adjusting the default output for certain
Numpy functions.

:Author: Samuel Farrens <samuel.farrens@cea.fr>

"""

import numpy as np


def rotate(input_data):
    """Rotate.

    This method rotates an input numpy array by 180 degrees.

    Parameters
    ----------
    input_data : numpy.ndarray
        Input data array (at least 2D)

    Returns
    -------
    numpy.ndarray
        Rotated data

    Notes
    -----
    Adjustment to numpy.rot90

    Examples
    --------
    >>> import numpy as np
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


    See Also
    --------
    numpy.rot90 : base function

    """
    return np.rot90(input_data, 2)


def rotate_stack(input_data):
    """Rotate stack.

    This method rotates each array in a stack of arrays by 180 degrees.

    Parameters
    ----------
    input_data : numpy.ndarray
        Input data array (at least 3D)

    Returns
    -------
    numpy.ndarray
        Rotated data

    Examples
    --------
    >>> import numpy as np
    >>> from modopt.base.np_adjust import rotate_stack
    >>> x = np.arange(18).reshape((2, 3, 3))
    >>> x
    array([[[ 0,  1,  2],
            [ 3,  4,  5],
            [ 6,  7,  8]],
    <BLANKLINE>
           [[ 9, 10, 11],
            [12, 13, 14],
            [15, 16, 17]]])
    >>> rotate_stack(x)
    array([[[ 8,  7,  6],
            [ 5,  4,  3],
            [ 2,  1,  0]],
    <BLANKLINE>
           [[17, 16, 15],
            [14, 13, 12],
            [11, 10,  9]]])

    See Also
    --------
    rotate : looped function

    """
    return np.array([rotate(array) for array in input_data])


def pad2d(input_data, padding):
    """Pad array.

    This method pads an input numpy array with zeros in all directions.

    Parameters
    ----------
    input_data : numpy.ndarray
        Input data array (at least 2D)
    padding : int or tuple
        Amount of padding in x and y directions, respectively

    Returns
    -------
    numpy.ndarray
        Padded data

    Raises
    ------
    ValueError
        For

    Notes
    -----
    Adjustment to numpy.pad()

    Examples
    --------
    >>> import numpy as np
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

    See Also
    --------
    numpy.pad : base function
    """
    input_data = np.array(input_data)

    if isinstance(padding, int):
        padding = np.array([padding])
    elif isinstance(padding, (tuple, list)):
        padding = np.array(padding)
    elif not isinstance(padding, np.ndarray):
        raise ValueError(
            'Padding must be an integer or a tuple (or list, np.ndarray) '
            + 'of itegers',
        )

    if padding.size == 1:
        padding = np.repeat(padding, 2)

    pad_x = (padding[0], padding[0])
    pad_y = (padding[1], padding[1])

    return np.pad(input_data, (pad_x, pad_y), 'constant')


def ftr(input_data):
    """Fancy transpose right.

    Apply fancy_transpose() to data with roll=1.

    Parameters
    ----------
    input_data : numpy.ndarray
        Input data array

    Returns
    -------
    numpy.ndarray
        Transposed data

    See Also
    --------
    fancy_transpose : base function

    """
    return fancy_transpose(input_data)


def ftl(input_data):
    """Fancy transpose left.

    Apply fancy_transpose() to data with roll=-1.

    Parameters
    ----------
    input_data : numpy.ndarray
        Input data array

    Returns
    -------
    numpy.ndarray
        Transposed data

    See Also
    --------
    fancy_transpose : base function

    """
    return fancy_transpose(input_data, -1)


def fancy_transpose(input_data, roll=1):
    """Fancy transpose.

    This method transposes a multidimensional matrix.

    Parameters
    ----------
    input_data : numpy.ndarray
        Input data array
    roll : int
        Roll direction and amount (default is ``1``)

    Returns
    -------
    numpy.ndarray
        Transposed data

    Notes
    -----
    Adjustment to numpy.transpose

    Examples
    --------
    >>> import numpy as np
    >>> from modopt.base.np_adjust import fancy_transpose
    >>> x = np.arange(27).reshape(3, 3, 3)
    >>> x
    array([[[ 0,  1,  2],
            [ 3,  4,  5],
            [ 6,  7,  8]],
    <BLANKLINE>
           [[ 9, 10, 11],
            [12, 13, 14],
            [15, 16, 17]],
    <BLANKLINE>
           [[18, 19, 20],
            [21, 22, 23],
            [24, 25, 26]]])
    >>> fancy_transpose(x)
    array([[[ 0,  3,  6],
            [ 9, 12, 15],
            [18, 21, 24]],
    <BLANKLINE>
           [[ 1,  4,  7],
            [10, 13, 16],
            [19, 22, 25]],
    <BLANKLINE>
           [[ 2,  5,  8],
            [11, 14, 17],
            [20, 23, 26]]])
    >>> fancy_transpose(x, roll=-1)
    array([[[ 0,  9, 18],
            [ 1, 10, 19],
            [ 2, 11, 20]],
    <BLANKLINE>
           [[ 3, 12, 21],
            [ 4, 13, 22],
            [ 5, 14, 23]],
    <BLANKLINE>
           [[ 6, 15, 24],
            [ 7, 16, 25],
            [ 8, 17, 26]]])

    See Also
    --------
    numpy.transpose : base function

    """
    axis_roll = np.roll(np.arange(input_data.ndim), roll)

    return np.transpose(input_data, axes=axis_roll)
