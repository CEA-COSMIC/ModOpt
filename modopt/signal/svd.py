# -*- coding: utf-8 -*-

"""SVD ROUTINES

This module contains methods for thresholding singular values.

:Author: Samuel Farrens <samuel.farrens@cea.fr>

"""

from __future__ import division
from builtins import zip
import numpy as np
from scipy.linalg import svd
from modopt.math.convolve import convolve
from modopt.signal.noise import thresh
from modopt.base.transform import cube2matrix, matrix2cube
from modopt.interface.errors import warn


def find_n_pc(u, factor=0.5):
    """Find number of principal components

    This method finds the minimum number of principal components required

    Parameters
    ----------
    u : np.ndarray
        Left singular vector of the original data
    factor : float, optional
        Factor for testing the auto correlation (default is '0.5')

    Returns
    -------
    int number of principal components

    Examples
    --------
    >>> from scipy.linalg import svd
    >>> from modopt.signal.svd import find_n_pc
    >>> x = np.arange(18).reshape(9, 2).astype(float)
    >>> find_n_pc(svd(x)[0])
    array([3])

    """

    if np.sqrt(u.shape[0]) % 1:
        raise ValueError('Invalid left singular value. The size of the first '
                         'dimenion of u must be perfect square.')

    # Get the shape of the array
    array_shape = np.repeat(np.int(np.sqrt(u.shape[0])), 2)

    # Find the auto correlation of the left singular vector.
    u_auto = [convolve(a.reshape(array_shape),
              np.rot90(a.reshape(array_shape), 2)) for a in u.T]

    # Return the required number of principal components.
    return np.sum([(a[tuple(zip(array_shape // 2))] ** 2 <= factor *
                   np.sum(a ** 2)) for a in u_auto])


def calculate_svd(data):
    """Calculate Singular Value Decomposition

    This method calculates the Singular Value Decomposition (SVD) of the input
    data using SciPy.

    Parameters
    ----------
    data : np.ndarray
        Input data array, 2D matrix

    Returns
    -------
    tuple of left singular vector, singular values and right singular vector

    Raises
    ------
    TypeError
        For invalid data type

    """

    if (not isinstance(data, np.ndarray)) or (data.ndim != 2):
        raise TypeError('Input data must be a 2D np.ndarray.')

    return svd(data, check_finite=False, lapack_driver='gesvd',
               full_matrices=False)


def svd_thresh(data, threshold=None, n_pc=None, thresh_type='hard'):
    r"""Threshold the singular values

    This method thresholds the input data using singular value decomposition

    Parameters
    ----------
    data : np.ndarray
        Input data array, 2D matrix
    threshold : float or np.ndarray, optional
        Threshold value(s)
    n_pc : int or str, optional
        Number of principal components, specify an integer value or 'all'
    threshold_type : str {'hard', 'soft'}, optional
        Type of thresholding (default is 'hard')

    Returns
    -------
    np.ndarray thresholded data

    Raises
    ------
    ValueError
        For invalid n_pc value

    Examples
    --------
    >>> from modopt.signal.svd import svd_thresh
    >>> x = np.arange(18).reshape(9, 2).astype(float)
    >>> svd_thresh(x, n_pc=1)
    array([[  0.49815487,   0.54291537],
           [  2.40863386,   2.62505584],
           [  4.31911286,   4.70719631],
           [  6.22959185,   6.78933678],
           [  8.14007085,   8.87147725],
           [ 10.05054985,  10.95361772],
           [ 11.96102884,  13.03575819],
           [ 13.87150784,  15.11789866],
           [ 15.78198684,  17.20003913]])

    """

    if ((not isinstance(n_pc, (int, str, type(None)))) or
            (isinstance(n_pc, int) and n_pc <= 0) or
            (isinstance(n_pc, str) and n_pc != 'all')):
        raise ValueError('Invalid value for "n_pc", specify a positive '
                         'integer value or "all"')

    # Get SVD of input data.
    u, s, v = calculate_svd(data)

    # Find the threshold if not provided.
    if isinstance(threshold, type(None)):

        # Find the required number of principal components if not specified.
        if isinstance(n_pc, type(None)):
            n_pc = find_n_pc(u, factor=0.1)

        # If the number of PCs is too large use all of the singular values.
        if ((isinstance(n_pc, int) and n_pc >= s.size) or
                (isinstance(n_pc, str) and n_pc == 'all')):
            n_pc = s.size
            warn('Using all singular values.')

        threshold = s[n_pc - 1]

    # Threshold the singular values.
    s_new = thresh(s, threshold, thresh_type)

    if np.all(s_new == s):
        warn('No change to singular values.')

    # Diagonalize the svd
    s_new = np.diag(s_new)

    # Return the thresholded data.
    return np.dot(u, np.dot(s_new, v))


def svd_thresh_coef(data, operator, threshold, thresh_type='hard'):
    """Threshold the singular values coefficients

    This method thresholds the input data using singular value decomposition

    Parameters
    ----------
    data : np.ndarray
        Input data array, 2D matrix
    operator : class
        Operator class instance
    threshold : float or np.ndarray
        Threshold value(s)
    threshold_type : str {'hard', 'soft'}
        Type of noise to be added (default is 'hard')

    Returns
    -------
    np.ndarray thresholded data

    Raises
    ------
    ValueError
        For invalid string entry for n_pc

    """

    if not callable(operator):
        raise TypeError('Operator must be a callable function.')

    # Get SVD of data matrix
    u, s, v = calculate_svd(data)

    # Diagnalise s
    s = np.diag(s)

    # Compute coefficients
    a = np.dot(s, v)

    # Get the shape of the array
    array_shape = np.repeat(np.int(np.sqrt(u.shape[0])), 2)

    # Compute threshold matrix.
    ti = np.array([np.linalg.norm(x) for x in
                   operator(matrix2cube(u, array_shape))])
    threshold *= np.repeat(ti, a.shape[1]).reshape(a.shape)

    # Threshold coefficients.
    a_new = thresh(a, threshold, thresh_type)

    # Return the thresholded image.
    return np.dot(u, a_new)
