# -*- coding: utf-8 -*-

"""INTEGRATION ROUTINES

This module contains methods for integration.

:Author: Samuel Farrens <samuel.farrens@gmail.com>

:Version: 1.0

:Date: 03/04/2017

"""


import numpy as np
from scipy.integrate import quad


def integrate(func, lim_low, lim_up, *args):
    """Integrate

    This method integrates a given function, which has N additional arguments,
    between the specified limits.

    Parameters
    ----------
    func : function
        Function to be integrated
    lim_low : float
        Lower limit
    lim_up : float
        Upper limit

    Returns
    -------
    Result of the definite integral

    """

    return quad(func, lim_low, lim_up, args=args)[0]


def vintegrate(func, lim_low, lim_up, *args):
    """Vectorised integration

    This method implements a vectorised version of integrate().

    Parameters
    ----------
    func : function
        Function to be integrated
    lim_low : float
        Lower limit
    lim_up : float
        Upper limit

    Returns
    -------
    Result of the definite integral

    """

    v_integ = np.vectorize(integrate)

    return v_integ(func, lim_low, lim_up, *args)
