# -*- coding: utf-8 -*-

"""ANGLE HANDLING ROUTINES

This module contains methods for handing angles and trigonometry.

:Author: Samuel Farrens <samuel.farrens@gmail.com>

:Version: 1.1

:Date: 20/10/2017

"""

from __future__ import division
import numpy as np


def deg2rad(angle):
    r"""Degrees to radians

    This method converts the angle from degrees to radians.

    Parameters
    ----------
    angle : float or np.ndarray
        Input angle in degrees

    Returns
    -------
    float angle in radians or np.ndarray of angles

    Examples
    --------
    >>> from modopt.math.angle import deg2rad
    >>> deg2rad(45.)
    0.7853981633974483

    Notes
    -----
    Implements the following equation:

    .. math::
        \mathrm{radians} = \mathrm{degrees} \times \frac{\pi}{180}

    """

    return angle * np.pi / 180.0


def rad2deg(angle):
    r"""Radians to degrees

    This method converts the angle from radians to degrees.

    Parameters
    ----------
    angle : float or np.ndarray
        Input angle in radians

    Returns
    -------
    float angle in degrees or np.ndarray of angles

    Examples
    --------
    >>> from modopt.math.angle import deg2rad
    >>> rad2deg(1.)
    57.29577951308232

    Notes
    -----
    Implements the following equation:

    .. math::
        \mathrm{degrees} = \mathrm{radians} \times \frac{180}{\pi}

    """

    return angle * 180.0 / np.pi


def ang_sep(point1, point2):
    r"""Angular separation

    This method calculates the angular separation in degrees between two
    points.

    Parameters
    ----------
    point1 : tuple
        Angular position of point 1 in degrees
    point1 : tuple
        Angular position of point 2 in degrees

    Returns
    -------
    float angular separation in degrees

    Examples
    --------
    >>> from modopt.math.angle import ang_sep
    >>> ang_sep((30.0, 0.0), (47.0, 10.0))
    19.647958606833164

    Notes
    -----
    Implements the following equation:

    .. math::
        \theta = \cos^{-1}[\sin(\delta_1)\sin(\delta_2)+
        \cos(\delta_1)\cos(\delta_2)\cos(\alpha_1-\alpha_2)]

    See https://en.wikipedia.org/wiki/Angular_distance

    """

    dist = np.around(np.sin(deg2rad(point1[1])) * np.sin(deg2rad(point2[1])) +
                     np.cos(deg2rad(point1[1])) * np.cos(deg2rad(point2[1])) *
                     np.cos(deg2rad(point1[0]) - deg2rad(point2[0])), 10)

    return rad2deg(np.array(np.arccos(dist)))
