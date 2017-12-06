# -*- coding: utf-8 -*-

"""MISCELLANOUS MATH ROUTINES

This module contains methods for various mathematical operations.

:Author: Samuel Farrens <samuel.farrens@gmail.com>

:Version: 1.1

:Date: 20/10/2017

"""

from __future__ import division
import numpy as np


def factor(n):
    """Factors of n

    This method finds factors of a number (n).

    Parameters
    ----------
    n : int
        Whole number

    Returns
    -------
    np.ndarray factors of n

    """

    factors = set()

    for x in range(1, int(np.sqrt(n)) + 1):
        if n % x == 0:
            factors.add(x)
            factors.add(n // x)

    return np.array(sorted(factors))


def mfactor(n):
    """Middle factors of n

    This method finds the middle factor(s) of a number (n).

    Parameters
    ----------
    n : int
        Whole number

    Returns
    -------
    np.ndarray middle factors of n

    """

    f = factor(n)

    if f.size % 2:
        return np.repeat(f[f.size // 2], 2)

    else:
        return f[f.size // 2 - 1:f.size // 2 + 1]


def k_val(n, L):
    """Spatial frequency

    This method returns k-values in the range L.

    Parameters
    ----------
    n : float
        Number
    limit : float
        Limit

    Returns
    -------
    float k value

    TODO
    ----
    Add equation and example

    """

    return ((2.0 * np.pi / limit) * np.array(range(n / 2.0) +
            range(-n / 2.0, 0.0)))


def fourier_derivative(func, k, order):
    """Fourier derivative

    This method returns the derivative of the specified function to the given
    order.

    Parameters
    ----------
    func : function
        Function
    k : float
        k-value
    order : int
        Oder of derivative

    Returns
    -------
        Float derivative

    TODO
    ----
    Add equation and example

    """

    return np.real(np.fft.ifft((1.j * k) ** order * np.fft.fft(func)))
