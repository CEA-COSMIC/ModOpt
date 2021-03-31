# -*- coding: utf-8 -*-

"""SVD ROUTINES.

This module contains methods for thresholding singular values.

:Author: Samuel Farrens <samuel.farrens@cea.fr>

"""

import numpy as np
from scipy.linalg import svd

from modopt.base.transform import matrix2cube
from modopt.interface.errors import warn
from modopt.math.convolve import convolve
from modopt.signal.noise import thresh


def find_n_pc(u_vec, factor=0.5):
    """Find number of principal components.

    This method finds the minimum number of principal components required.

    Parameters
    ----------
    u_vec : numpy.ndarray
        Left singular vector of the original data
    factor : float, optional
        Factor for testing the auto correlation (default is ``0.5``)

    Returns
    -------
    int
        Number of principal components

    Raises
    ------
    ValueError
        Invalid left singular vector

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import svd
    >>> from modopt.signal.svd import find_n_pc
    >>> x = np.arange(18).reshape(9, 2).astype(float)
    >>> find_n_pc(svd(x)[0])
    1

    """
    if np.sqrt(u_vec.shape[0]) % 1:
        raise ValueError(
            'Invalid left singular vector. The size of the first '
            + 'dimenion of ``u_vec`` must be perfect square.',
        )

    # Get the shape of the array
    array_shape = np.repeat(np.int(np.sqrt(u_vec.shape[0])), 2)

    # Find the auto correlation of the left singular vector.
    u_auto = [
        convolve(
            elem.reshape(array_shape),
            np.rot90(elem.reshape(array_shape), 2),
        )
        for elem in u_vec.T
    ]

    # Return the required number of principal components.
    return np.sum([
        (
            u_val[tuple(zip(array_shape // 2))] ** 2 <= factor
            * np.sum(u_val ** 2),
        )
        for u_val in u_auto
    ])


def calculate_svd(input_data):
    """Calculate Singular Value Decomposition.

    This method calculates the Singular Value Decomposition (SVD) of the input
    data using SciPy.

    Parameters
    ----------
    input_data : numpy.ndarray
        Input data array, 2D matrix

    Returns
    -------
    tuple
        Left singular vector, singular values and right singular vector

    Raises
    ------
    TypeError
        For invalid data type

    """
    if (not isinstance(input_data, np.ndarray)) or (input_data.ndim != 2):
        raise TypeError('Input data must be a 2D np.ndarray.')

    return svd(
        input_data,
        check_finite=False,
        lapack_driver='gesvd',
        full_matrices=False,
    )


def svd_thresh(input_data, threshold=None, n_pc=None, thresh_type='hard'):
    """Threshold the singular values.

    This method thresholds the input data using singular value decomposition.

    Parameters
    ----------
    input_data : numpy.ndarray
        Input data array, 2D matrix
    threshold : float or numpy.ndarray, optional
        Threshold value(s) (default is ``None``)
    n_pc : int or str, optional
        Number of principal components, specify an integer value or 'all'
        (default is ``None``)
    thresh_type : {'hard', 'soft'}, optional
        Type of thresholding (default is 'hard')

    Returns
    -------
    numpy.ndarray
        Thresholded data

    Raises
    ------
    ValueError
        For invalid n_pc value

    Examples
    --------
    >>> import numpy as np
    >>> from modopt.signal.svd import svd_thresh
    >>> x = np.arange(18).reshape(9, 2).astype(float)
    >>> svd_thresh(x, n_pc=1)
    array([[ 0.49815487,  0.54291537],
           [ 2.40863386,  2.62505584],
           [ 4.31911286,  4.70719631],
           [ 6.22959185,  6.78933678],
           [ 8.14007085,  8.87147725],
           [10.05054985, 10.95361772],
           [11.96102884, 13.03575819],
           [13.87150784, 15.11789866],
           [15.78198684, 17.20003913]])

    """
    less_than_zero = isinstance(n_pc, int) and n_pc <= 0
    str_not_all = isinstance(n_pc, str) and n_pc != 'all'

    if (
        (not isinstance(n_pc, (int, str, type(None))))
        or less_than_zero
        or str_not_all
    ):
        raise ValueError(
            'Invalid value for "n_pc", specify a positive integer value or '
            + '"all"',
        )

    # Get SVD of input data.
    u_vec, s_values, v_vec = calculate_svd(input_data)

    # Find the threshold if not provided.
    if isinstance(threshold, type(None)):
        # Find the required number of principal components if not specified.
        if isinstance(n_pc, type(None)):
            n_pc = find_n_pc(u_vec, factor=0.1)
            print('xxxx', n_pc, u_vec)

        # If the number of PCs is too large use all of the singular values.
        if (
            (isinstance(n_pc, int) and n_pc >= s_values.size)
            or (isinstance(n_pc, str) and n_pc == 'all')
        ):
            n_pc = s_values.size
            warn('Using all singular values.')

        threshold = s_values[n_pc - 1]

    # Threshold the singular values.
    s_new = thresh(s_values, threshold, thresh_type)

    if np.all(s_new == s_values):
        warn('No change to singular values.')

    # Diagonalize the svd
    s_new = np.diag(s_new)

    # Return the thresholded data.
    return np.dot(u_vec, np.dot(s_new, v_vec))


def svd_thresh_coef(input_data, operator, threshold, thresh_type='hard'):
    """Threshold the singular values coefficients.

    This method thresholds the input data using singular value decomposition

    Parameters
    ----------
    input_data : numpy.ndarray
        Input data array, 2D matrix
    operator : class
        Operator class instance
    threshold : float or numpy.ndarray
        Threshold value(s)
    thresh_type : {'hard', 'soft'}
        Type of noise to be added (default is 'hard')

    Returns
    -------
    numpy.ndarray
        Thresholded data

    Raises
    ------
    TypeError
        If operator not callable

    """
    if not callable(operator):
        raise TypeError('Operator must be a callable function.')

    # Get SVD of data matrix
    u_vec, s_values, v_vec = calculate_svd(input_data)

    # Diagnalise s
    s_values = np.diag(s_values)

    # Compute coefficients
    a_matrix = np.dot(s_values, v_vec)

    # Get the shape of the array
    array_shape = np.repeat(np.int(np.sqrt(u_vec.shape[0])), 2)

    # Compute threshold matrix.
    ti = np.array([
        np.linalg.norm(elem)
        for elem in operator(matrix2cube(u_vec, array_shape))
    ])
    threshold *= np.repeat(ti, a_matrix.shape[1]).reshape(a_matrix.shape)

    # Threshold coefficients.
    a_new = thresh(a_matrix, threshold, thresh_type)

    # Return the thresholded image.
    return np.dot(u_vec, a_new)
