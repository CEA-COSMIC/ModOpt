# -*- coding: utf-8 -*-

"""REGRESSION ROUTINES

This module contains methods for linear regression.

:Author: Samuel Farrens <samuel.farrens@gmail.com>

:Version: 1.0

:Date: 04/04/2017

"""

from builtins import zip
import numpy as np
from scipy.odr import *


def linear_fit(B, x):
    r"""Linear fit

    This method defines the equation of a straight line.

    Parameters
    ----------
    B : tuple
        Slope (m) and intercept (b) of the line.
    x : list or np.ndarray
        The 1D data vector

    Returns
    -------
    np.ndarray 1D array of corresponding y values

    NOTES
    -----
    This equation of a stright line is given by

    .. math::

        y = mx + b

    """

    return B[0] * np.array(x) + B[1]


def polynomial(x, a):
    r"""Polynomial

    This method defines the equation of a polynomial line.

    Parameters
    ----------
    x : list or np.ndarray
        The 1D data vector
    a : list or np.ndarray
        The 1D polynomial coefficient vector

    Returns
    -------
    np.ndarray 1D array of corresponding y values

    NOTES
    -----
    This equation of a stright line is given by

    .. math::

        y = a_0 + a_1x + a_2x^2 + \dots + a_kx^k

    """

    a = np.array(a)
    x = np.array(x)

    return sum([(a_i * x ** n) for a_i, n in zip(a, range(a.size))])


def polynomial_fit(x, y, k=1):
    """Polynomial fit

    This method finds the coefficients for a polynomial line fit to the input
    data using least squares.

    Parameters
    ----------
    x : list or np.ndarray
        The 1D independent data vector
    y : list or np.ndarray
        The 1D dependent data vector
    k : int, optional
        Number of degrees of freedom. Default (k=1)

    Returns
    -------
    np.ndarray 1D array of coefficients a

    """

    y = np.array(y)

    return least_squares(x_matrix(x, k), y)


def least_squares(X, y):
    """Least squares

    This method performs an analytical least squares regression. Returns the
    values of the coefficients, a, given the input matrix X and the
    corresponding y values.

    Parameters
    ----------
    X : np.ndarray
        The 2D independent data matrix.
    y : np.ndarray
        The 1D dependent data vector

    Returns
    -------
    np.ndarray 1D array of coefficients a

    Raises
    ------
    ValueError
        If inputs are not numpy arrays

    ToDo
    ----
    Add equation and example

    """

    if not np.all([isinstance(i, np.ndarray) for i in (X, y)]):
        raise ValueError('Inputs must be numpy arrays.')

    return np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)


def x_matrix(x, k):
    """Define X matrix

    This method defines the matrix X for a given vector x corresponding to a
    polynomial with k degrees of freedom.

    Parameters
    ----------
    x : list or np.ndarray
        The 1D independent data vector
    k : int
        Number of degrees of freedom

    Returns
    -------
    np.ndarray the 2D independent variable matrix X

    """

    x = np.array(x)

    return np.vstack([x ** n for n in range(k + 1)]).T


def fit_odr(x, y, xerr, yerr, fit):
    """Fit via ODR

    This method performs an orthogonal distance regression fit.

    Parameters
    ----------
    x : list or np.ndarray
        The 1D independent data vector
    y : list or np.ndarray
        The 1D dependent data vector
    x_err : list or np.ndarray
        1D data vector of x value errors
    y_err : list or np.ndarray
        1D data vector of y value errors
    fit : function
        Fitting function

    Returns
    -------
    tuple best fit parameters

    """

    model = Model(fit)
    r_data = RealData(x, y, sx=xerr, sy=yerr)
    odr = ODR(r_data, model, beta0=[1.0, 2.0])
    odr_out = odr.run()

    return odr_out.beta
