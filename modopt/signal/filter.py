# -*- coding: utf-8 -*-

"""FILTER ROUTINES

This module contains methods for distance measurements in cosmology.

:Author: Samuel Farrens <samuel.farrens@gmail.com>

:Version: 1.1

:Date: 20/10/2017

"""

from __future__ import division
import numpy as np
from modopt.base.types import check_float


def Gaussian_filter(x, sigma, norm=True):
    """Gaussian filter

    This method implements a Gaussian filter.

    Parameters
    ----------
    x : float
        Input data point
    sigma : float
        Standard deviation (filter scale)
    norm : bool
        Option to return normalised data. Default (norm=True)

    Returns
    -------
    float Gaussian filtered data point

    """

    x = check_float(x)
    sigma = check_float(sigma)

    val = np.exp(-0.5 * (x / sigma) ** 2)

    if norm:
        return val / (np.sqrt(2 * np.pi) * sigma)

    else:
        return val


def mex_hat(x, sigma):
    """Mexican hat

    This method implements a Mexican hat (or Ricker) wavelet.

    Parameters
    ----------
    x : float
        Input data point
    sigma : float
        Standard deviation (filter scale)

    Returns
    -------
    float Mexican hat filtered data point

    """

    x = check_float(x)
    sigma = check_float(sigma)

    xs = (x / sigma) ** 2
    val = 2 * (3 * sigma) ** -0.5 * np.pi ** -0.25

    return val * (1 - xs) * np.exp(-0.5 * xs)


def mex_hat_dir(x, y, sigma):
    """Directional Mexican hat

    This method implements a directional Mexican hat (or Ricker) wavelet.

    Parameters
    ----------
    x : float
        Input data point for Gaussian
    y : float
        Input data point for Mexican hat
    sigma : float
        Standard deviation (filter scale)

    Returns
    -------
    float directional Mexican hat filtered data point

    """

    return -0.5 * (x / sigma) ** 2 * mex_hat(y, sigma)
