# -*- coding: utf-8 -*-

"""STATISTICS ROUTINES

This module contains methods for basic statistics.

:Author: Samuel Farrens <samuel.farrens@gmail.com>

:Version: 1.2

:Date: 20/10/2017

"""

from __future__ import division
from builtins import zip
import numpy as np
from scipy.stats import chi2
from astropy.convolution import Gaussian2DKernel


def chi2_gof(data_obs, data_exp, sigma, ddof=1):
    """Chi-squared goodness-of-fit

    This method tests the chi^2 goodness of fit.

    Parameters
    ----------
    data_obs : np.ndarray
        Observed data array
    data_exp : np.ndarray
        Expected data array
    sigma : float
        Expected data error
    ddof : input
        Delta degrees of freedom. Default (ddof = 1).

    Returns
    -------
    tuple of floats chi-squared and P values

    """

    chi2 = np.sum(((data_obs - data_exp) / sigma) ** 2)
    p_val = chi2.cdf(chi2, len(data_obs) - ddof)

    return chi2, p_val


def gaussian(point, mean, sigma, amplitude=None):
    """Gaussian distribution

    Method under development...

    """

    if isinstance(amplitude, type(None)):
        amplitude = 1

    val = np.array([((x - mu) / sig) ** 2 for x, mu, sig in
                   zip(point, mean, sigma)])

    return amplitude * np.exp(-0.5 * val)


def gaussian_kernel(data_shape, sigma, norm='max'):
    """Gaussian kernel

    This method produces a Gaussian kerenal of a specified size and dispersion

    Parameters
    ----------
    data_shape : tuple
        Desiered shape of the kernel
    sigma : float
        Standard deviation of the kernel
    norm : str {'max', 'sum'}, optional
        Normalisation of the kerenl (options are 'max' or 'sum')

    Returns
    -------
    np.ndarray kernel

    """

    kernel = np.array(Gaussian2DKernel(sigma, x_size=data_shape[1],
                      y_size=data_shape[0]))

    if norm is 'max':
        return kernel / np.max(kernel)

    elif norm is 'sum':
        return kernel / np.sum(kernel)

    else:
        return kernel


def mad(data):
    r"""Median absolute deviation

    This method calculates the median absolute deviation of the input data.

    Parameters
    ----------
    data : np.ndarray
        Input data array

    Returns
    -------
    float MAD value

    Notes
    -----
    The MAD is calculated as follows:

    .. math::

        \mathrm{MAD} = \mathrm{median}\left(|X_i - \mathrm{median}(X)|\right)

    """

    return np.median(np.abs(data - np.median(data)))


def mse(data1, data2):
    """Mean Squared Error

    This method returns the Mean Squared Error (MSE) between two data sets.

    Parameters
    ----------
    data1 : np.ndarray
        First data set
    data2 : np.ndarray
        Second data set

    """

    return np.mean((data1 - data2) ** 2)


def psnr2(image, noisy_image, max_pix=255):
    r"""Peak Signal-to-Noise Ratio

    This method calculates the PSNR between an image and a noisy version
    of that image

    Parameters
    ----------
    image : np.ndarray
        Input image, 2D array
    noisy_image : np.ndarray
        Noisy image, 2D array
    max_pix : int
        Maximum number of pixels. Default (max_pix=255)

    Returns
    -------
    float PSNR value

    Notes
    -----
    Implements PSNR equation on
    https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    .. math::

        \mathrm{PSNR} = 20\log_{10}(\mathrm{MAX}_I - 10\log_{10}(\mathrm{MSE}))

    """

    return (20 * np.log10(max_pix) - 10 *
            np.log10(mse(image, noisy_image)))


def psnr(image, recovered_image):
    """Peak Signal-to-Noise Ratio

    This method calculates the PSNR between an image and the recovered version
    of that image

    Parameters
    ----------
    image : np.ndarray
        Input image, 2D array
    recovered_image : np.ndarray
        Recovered image, 2D array

    Returns
    -------
    float PSNR value

    Notes
    -----
    Implements eq.3.7 from _[S2010]

    """

    return (20 * np.log10((image.shape[0] * np.abs(np.max(image) -
            np.min(image))) / np.linalg.norm(image - recovered_image)))


def psnr_stack(images, recoverd_images, metric=np.mean):
    """Peak Signa-to-Noise for stack of images

    This method calculates the PSNRs for a stack of images and the
    corresponding recovered images. By default the metod returns the mean
    value of the PSNRs, but any other metric can be used.

    Parameters
    ----------
    images : np.ndarray
        Stack of images, 3D array
    recovered_images : np.ndarray
        Stack of recovered images, 3D array
    metric : function
        The desired metric to be applied to the PSNR values (default is
        'np.mean')

    Returns
    -------
    float metric result of PSNR values

    Raises
    ------
    ValueError
        For invalid input data dimensions

    """

    if images.ndim != 3 or recoverd_images.ndim != 3:
        raise ValueError('Input data must be a 3D np.ndarray')

    return metric([psnr(i, j) for i, j in zip(images, recoverd_images)])


def sigma_mad(data):
    r"""Standard deviation from MAD

    This method calculates the standard deviation of the input data from the
    MAD.

    Parameters
    ----------
    data : np.ndarray
        Input data array

    Returns
    -------
    float sigma value

    Notes
    -----
    This function can be used for estimating the standeviation of the noise in
    imgaes.

    Sigma is calculated as follows:

    .. math::

        \sigma = 1.4826 \mathrm{MAD}(X)

    """

    return 1.4826 * mad(data)
