# -*- coding: utf-8 -*-

"""QUALITY ASSESSMENT ROUTINES

This module contains methods and classes for assessing the quality of image
reconstructions.

:Author: Samuel Farrens <samuel.farrens@gmail.com>

:Version: 1.2

:Date: 20/10/2017

Notes
-----
Some of the methods in this module are based on work by Fred Ngole.

"""

from __future__ import division
import numpy as np
from modopt.image.shape import ellipticity_atoms


def nmse(image1, image2, metric=np.mean):
    r"""Normalised Mean Square Error

    This method computes the NMSE of two input images, or the result of the
    input metric on a stack of input images.

    Parameters
    ----------
    image1 : np.ndarray
        First image (or stack of images) to be analysed (original image)
    image2 : np.ndarray
        Second image (or stack of images) to be analysed (reconstructed image)
    metric : function
        Metric to be apllied to NMSE results (default is 'np.mean')

    Returns
    -------
    float NMSE value or metric value(s)

    Raises
    ------
    ValueError
        For invalid input data dimensions

    See Also
    --------
    e_error : ellipticity error

    Notes
    -----
    This method implements the following equation:

        - Equations from [NS2016]_ sec 4.1:

        .. math::

            \text{NMSE} = \frac{1}{D}\sum_{i=1}^D
                \frac{\|\hat{\text{Im}}_i - \text{Im}_i\|_2^2}
                {\|\text{Im}_i\|_2^2}

    Examples
    --------
    >>> from image.quality import nmse

    """

    if image1.ndim != image2.ndim:
        raise ValueError('Input images must have the same dimensions')

    if image1.ndim not in (2, 3):
        raise ValueError('Input data must be single image or stack of images')

    if image1.ndim == 2:

        return (np.linalg.norm(image2 - image1) ** 2 /
                np.linalg.norm(image1) ** 2)

    else:

        res = (np.array([np.linalg.norm(x) ** 2 for x in (image2 - image1)]) /
               np.array([np.linalg.norm(x) ** 2 for x in image1]))

        return metric(res)


def e_error(image1, image2, metric=np.mean):
    r"""Normalised Mean Square Error

    This method computes the ellipticity error of two input images, or the
    result of the input metric on the ellipticity values.

    Parameters
    ----------
    image1 : np.ndarray
        First image to be analysed (original image)
    image2 : np.ndarray
        Second image to be analysed (reconstructed image)
    metric : function
        Metric to be apllied to ellipticity error results (default is
        'np.mean')

    Returns
    -------
    float ellipticity error value or metric value(s)

    Raises
    ------
    ValueError
        For invalid input data dimensions

    See Also
    --------
    nmse : nmse error

    Notes
    -----
    This method implements the following equation:

        - Equations from [NS2016]_ sec 4.1:

        .. math::

            \text{E}_\gamma = \frac{1}{D}\sum_{i=1}^D
                \|\gamma(\text{Im}_i) - \gamma(\hat{\text{Im}}_i)\|_2

    Examples
    --------
    >>> from image.quality import e_error

    """

    if image1.ndim != image2.ndim:
        raise ValueError('Input images must have the same dimensions')

    if image1.ndim not in (2, 3):
        raise ValueError('Input data must be single image or stack of images')

    if image1.ndim == 2:

        return np.linalg.norm(ellipticity_atoms(image1) -
                              ellipticity_atoms(image2))

    else:

        diff = (np.array([ellipticity_atoms(x) for x in image1]) -
                np.array([ellipticity_atoms(x) for x in image2]))

        return metric([np.linalg.norm(x) for x in diff])
