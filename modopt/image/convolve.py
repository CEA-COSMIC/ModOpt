# -*- coding: utf-8 -*-

"""CONVOLUTION ROUTINES

This module contains methods for convolution.

:Author: Samuel Farrens <samuel.farrens@gmail.com>

:Version: 1.2

:Date: 23/10/2017

"""

from __future__ import division
from builtins import zip
import numpy as np
from scipy.signal import fftconvolve
from scipy.fftpack import fftn, ifftn, fftshift, ifftshift
from astropy.convolution import convolve_fft
from modopt.base.np_adjust import rotate, rotate_stack


def convolve_np(image, kernel):
    """Convolve with Numpy FFT

    This method convolves the input image with the input kernel

    Parameters
    ----------
    image : np.ndarray
        2D image array
    kernel : np.ndarray
        2D kernel array

    Returns
    -------
    np.ndarray 2D convolved image array

    """

    x = np.fft.fftshift(np.fft.fftn(image))
    y = np.fft.fftshift(np.fft.fftn(kernel))

    return np.real(np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(x * y))))


def deconvolve_np(image, kernel):
    """Deconvolve with Numpy FFT

    This method deconvolves the input image with the input kernel

    Parameters
    ----------
    image : np.ndarray
        2D image array
    kernel : np.ndarray
        2D kernel array

    Returns
    -------
    np.ndarray 2D deconvolved image array

    """

    x = np.fft.fftshift(np.fft.fftn(image))
    y = np.fft.fftshift(np.fft.fftn(kernel))

    return np.real(np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(x / y))))


def convolve(data, kernel, method='astropy'):
    """Convolve data with kernel

    This method convolves the input data with a given kernel using FFT and
    is the default convolution used for all routines

    Parameters
    ----------
    data : np.ndarray
        Input data array, normally a 2D image
    kernel : np.ndarray
        Input kernel array, normally a 2D kernel
    method : str {'astropy', 'scipy'}, optional
        Convolution method (default is 'astropy')

        'astropy':
            Uses the astropy.convolution.convolve_fft method provided in
            Astropy (http://www.astropy.org/)

        'scipy':
            Uses the scipy.signal.fftconvolve method provided in SciPy
            (https://www.scipy.org/)

    Returns
    -------
    np.ndarray convolved data

    Raises
    ------
    ValueError
        If `data` and `kernel` do not have the same number of dimensions
    ValueError
        If `method` is not 'astropy' or 'scipy'

    """

    if data.ndim != kernel.ndim:
        raise ValueError('Data and kernel must have the same dimensions.')

    if method not in ('astropy', 'scipy'):
        raise ValueError('Invalid method. Options are "astropy" or "scipy".')

    if method == 'astropy':
        return convolve_fft(data, kernel, boundary='wrap', crop=False,
                            nan_treatment='fill', normalize_kernel=False)

    elif method == 'scipy':
        return fftconvolve(data, kernel, mode='same')


def convolve_stack(data, kernel, rot_kernel=False, method='astropy'):
    """Convolve stack of data with stack of kernels

    This method convolves the input data with a given kernel using FFT and
    is the default convolution used for all routines

    Parameters
    ----------
    data : np.ndarray
        Input data array, normally a 2D image
    kernel : np.ndarray
        Input kernel array, normally a 2D kernel
    rot_kernel : bool
        Option to rotate kernels by 180 degrees
    method : str {'astropy', 'scipy'}, optional
        Convolution method (default is 'astropy')

    Returns
    -------
    np.ndarray convolved data

    """

    if rot_kernel:
        kernel = rotate_stack(kernel)

    return np.array([convolve(data_i, kernel_i, method=method) for data_i,
                    kernel_i in zip(data, kernel)])


def psf_convolve(data, psf, psf_rot=False, psf_type='fixed', method='astropy'):
    """Convolve data with PSF

    This method convolves an image with a PSF

    Parameters
    ----------
    data : np.ndarray
        Input data array, normally an array of 2D images
    psf : np.ndarray
        Input PSF array, normally either a single 2D PSF or an array of 2D
        PSFs
    psf_rot: bool
        Option to rotate PSF by 180 degrees
    psf_type : str {'fixed', 'obj_var'}, optional
        PSF type (default is 'fixed')
    method : str {'astropy', 'scipy'}, optional
        Convolution method (default is 'astropy')

        'fixed':
            The PSF is fixed, i.e. it is the same for each image

        'obj_var':
            The PSF is object variant, i.e. it is different for each image

    Returns
    -------
    np.ndarray convolved data

    Raises
    ------
    ValueError
        If `psf_type` is not 'fixed' or 'obj_var'

    """

    if psf_type not in ('fixed', 'obj_var'):
        raise ValueError('Invalid PSF type. Options are "fixed" or "obj_var"')

    if psf_rot and psf_type == 'fixed':
        psf = rotate(psf)

    elif psf_rot:
        psf = rotate_stack(psf)

    if psf_type == 'fixed':
        return np.array([convolve(data_i, psf, method=method) for data_i in
                        data])

    elif psf_type == 'obj_var':

        return convolve_stack(data, psf)


def pseudo_inverse(image, kernel, weight=None):
    """Pseudo inverse

    This method calculates the pseudo inverse of the input image for the given
    kernel using FFT

    Parameters
    ----------
    image : np.ndarray
        Input image, 2D array
    kernel : np.ndarray
        Input kernel, 2D array
    weight : np.ndarray, optional
        Optional weights, 2D array

    Returns
    -------
    np.ndarray result of the pseudo inverse

    """

    y_hat = fftshift(fftn(image))
    h_hat = fftshift(fftn(kernel))
    h_hat_star = np.conj(h_hat)

    res = ((h_hat_star * y_hat) / (h_hat_star * h_hat))

    if not isinstance(weight, type(None)):
        res *= weight

    return np.real(fftshift(ifftn(ifftshift(res))))
