# -*- coding: utf-8 -*-

"""SVD ROUTINES

This module contains methods for thresholding singular values.

:Author: Samuel Farrens <samuel.farrens@gmail.com>

:Version: 1.2

:Date: 20/10/2017

"""

from __future__ import division
from builtins import zip
import numpy as np
from scipy.linalg import svd, diagsvd
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
        Left singular vector
    factor : float, optional
        Factor for testing the auto correlation (default is '0.5')

    Returns
    -------
    int number of principal components

    """

    # Get the shape of the galaxy images.
    gal_shape = np.repeat(np.int(np.sqrt(u.shape[0])), 2)

    # Find the auto correlation of the left singular vector.
    u_auto = [convolve(a.reshape(gal_shape), np.rot90(a.reshape(gal_shape), 2))
              for a in u.T]

    # Return the required number of principal components.
    return np.sum(((a[list(zip(gal_shape // 2))] ** 2 <= factor *
                  np.sum(a ** 2)) for a in u_auto))


def svd_thresh(data, threshold=None, n_pc=None, thresh_type='hard'):
    """Threshold the singular values

    This method thresholds the input data using singular value decomposition

    Parameters
    ----------
    data : np.ndarray
        Input data array
    threshold : float, optional
        Threshold value
    n_pc : int or str, optional
        Number of principal components, specify an integer value or 'all'
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

    if isinstance(n_pc, str) and n_pc != 'all':
        raise ValueError('Invalid value for "n_pc", specify an integer value '
                         'or "all"')

    # Get SVD of input data.
    u, s, v = svd(data, check_finite=False, lapack_driver='gesvd')

    # Find the threshold if not provided.
    if isinstance(threshold, type(None)):

        # Find the required number of principal components if not specified.
        if isinstance(n_pc, type(None)):
            n_pc = find_n_pc(u, factor=0.1)

        # If the number of PCs is too large use all of the singular values.
        if n_pc >= s.size or n_pc == 'all':
            n_pc = s.size - 1
            warn('Using all singular values.')

        threshold = s[n_pc]

    # Remove noise from singular values.
    s_new = thresh(s, threshold, thresh_type)

    if np.all(s_new == s):
        warn('No change to singular values.')

    # Reshape the singular values to the shape of the input image.
    s_new = diagsvd(s_new, *data.shape)

    # Return the thresholded image.
    return np.dot(u, np.dot(s_new, v))


def svd_thresh_coef(data, operator, threshold, thresh_type='hard'):
    """Threshold the singular values coefficients

    This method thresholds the input data using singular value decomposition

    Parameters
    ----------
    data : np.ndarray
        Input data array
    operator : class
        Operator class instance
    threshold : float, optional
        Threshold value
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

    # Convert data cube to matrix.
    data_matrix = cube2matrix(data)

    # Get SVD of data matrix.
    u, s, v = np.linalg.svd(data_matrix, full_matrices=False)

    # Compute coefficients.
    a = np.dot(np.diag(s), v)

    # Compute threshold matrix.
    u_cube = matrix2cube(u, data.shape[1:])
    ti = np.array([np.linalg.norm(x) for x in operator(u_cube)])
    ti = np.repeat(ti, a.shape[1]).reshape(a.shape)
    threshold *= ti

    # Remove noise from coefficients.
    a_new = thresh(a, threshold, thresh_type)

    # Return the thresholded image.
    return np.dot(u, a_new)
