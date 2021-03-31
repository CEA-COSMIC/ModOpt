# -*- coding: utf-8 -*-

"""FILTER ROUTINES.

This module contains methods for distance measurements in cosmology.

:Author: Samuel Farrens <samuel.farrens@cea.fr>

"""

import numpy as np

from modopt.base.types import check_float


def gaussian_filter(data_point, sigma, norm=True):
    """Gaussian filter.

    This method implements a Gaussian filter.

    Parameters
    ----------
    data_point : float
        Input data point
    sigma : float
        Standard deviation (filter scale)
    norm : bool
        Option to return normalised data (default is ``True``)

    Returns
    -------
    float
        Gaussian filtered data point

    Examples
    --------
    >>> from modopt.signal.filter import gaussian_filter
    >>> gaussian_filter(1, 1)
    0.24197072451914337

    >>> gaussian_filter(1, 1, False)
    0.6065306597126334

    """
    data_point = check_float(data_point)
    sigma = check_float(sigma)

    numerator = np.exp(-0.5 * (data_point / sigma) ** 2)

    if norm:
        return numerator / (np.sqrt(2 * np.pi) * sigma)

    return numerator


def mex_hat(data_point, sigma):
    """Mexican hat.

    This method implements a Mexican hat (or Ricker) wavelet.

    Parameters
    ----------
    data_point : float
        Input data point
    sigma : float
        Standard deviation (filter scale)

    Returns
    -------
    float
        Mexican hat filtered data point

    Examples
    --------
    >>> from modopt.signal.filter import mex_hat
    >>> mex_hat(2, 1)
    -0.3521390522571337

    """
    data_point = check_float(data_point)
    sigma = check_float(sigma)

    xs = (data_point / sigma) ** 2
    factor = 2 * (3 * sigma) ** -0.5 * np.pi ** -0.25

    return factor * (1 - xs) * np.exp(-0.5 * xs)


def mex_hat_dir(data_gauss, data_mex, sigma):
    """Directional Mexican hat.

    This method implements a directional Mexican hat (or Ricker) wavelet.

    Parameters
    ----------
    data_gauss : float
        Input data point for Gaussian
    data_mex : float
        Input data point for Mexican hat
    sigma : float
        Standard deviation (filter scale)

    Returns
    -------
    float
        Directional Mexican hat filtered data point

    Examples
    --------
    >>> from modopt.signal.filter import mex_hat_dir
    >>> mex_hat_dir(1, 2, 1)
    0.17606952612856686

    """
    data_gauss = check_float(data_gauss)
    sigma = check_float(sigma)

    return -0.5 * (data_gauss / sigma) ** 2 * mex_hat(data_mex, sigma)
