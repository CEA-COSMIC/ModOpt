# -*- coding: utf-8 -*-

"""METRICS

This module contains classes of different metric functions for optimization.

:Author: Benoir Sarthou

"""

import numpy as np
from scipy.ndimage import uniform_filter, gaussian_filter
try:
    from skimage.measure import compare_ssim
except ImportError:  # pragma: no cover
    import_skimage = False
else:
    import_skimage = True


def min_max_normalize(img):
    """Centre and normalize a given array.

    Parameters:
    ----------
    img: np.ndarray

    """

    min_img = img.min()
    max_img = img.max()

    return (img - min_img) / (max_img - min_img)


def _preprocess_input(test, ref, mask=None):
    """Wrapper to the metric

    Parameters
    ----------
    ref : np.ndarray
        the reference image
    test : np.ndarray
        the tested image
    mask : np.ndarray, optional
        the mask for the ROI

    Notes
    -----
    Compute the metric only on magnetude.

    Returns
    -------
    ssim: float, the snr

    """

    test = np.abs(np.copy(test)).astype('float64')
    ref = np.abs(np.copy(ref)).astype('float64')
    test = min_max_normalize(test)
    ref = min_max_normalize(ref)

    if (not isinstance(mask, np.ndarray)) and (mask is not None):
        raise ValueError("mask should be None, or a np.ndarray,"
                         " got '{0}' instead.".format(mask))

    if mask is None:
        return test, ref, None

    return test, ref, mask


def ssim(test, ref, mask=None):
    """Structural Similarity (SSIM)

    Calculate the SSIM between a test image and a reference image.

    Parameters
    ----------
    ref : np.ndarray
        the reference image
    test : np.ndarray
        the tested image
    mask : np.ndarray, optional
        the mask for the ROI

    Notes
    -----
    Compute the metric only on magnetude.

    Returns
    -------
    ssim: float, the snr

    """

    if not import_skimage:  # pragma: no cover
        raise ImportError('Scikit-Image package not found')

    test, ref, mask = _preprocess_input(test, ref, mask)
    assim, ssim = compare_ssim(test, ref, full=True)

    if mask is None:
        return assim

    else:
        return (mask * ssim).sum() / mask.sum()


def snr(test, ref, mask=None):
    """Signal-to-Noise Ratio (SNR)

    Calculate the SNR between a test image and a reference image.

    Parameters
    ----------
    ref: np.ndarray
        the reference image
    test: np.ndarray
        the tested image
    mask: np.ndarray, optional
        the mask for the ROI

    Notes
    -----
    Compute the metric only on magnetude.

    Returns
    -------
    snr: float, the snr

    """

    test, ref, mask = _preprocess_input(test, ref, mask)

    if mask is not None:
        test = mask * test

    num = np.mean(np.square(test))
    deno = mse(test, ref)

    return 10.0 * np.log10(num / deno)


def psnr(test, ref, mask=None):
    """Peak Signal-to-Noise Ratio (PSNR)

    Calculate the PSNR between a test image and a reference image.

    Parameters
    ----------
    ref : np.ndarray
        the reference image
    test : np.ndarray
        the tested image
    mask : np.ndarray, optional
        the mask for the ROI

    Notes
    -----
    Compute the metric only on magnetude.

    Returns
    -------
    psnr: float, the psnr

    """

    test, ref, mask = _preprocess_input(test, ref, mask)

    if mask is not None:
        test = mask * test
        ref = mask * ref

    num = np.max(np.abs(test))
    deno = mse(test, ref)

    return 10.0 * np.log10(num / deno)


def mse(test, ref, mask=None):
    """Mean Squared Error (MSE)

    Calculate the MSE between a test image and a reference image.

    Parameters
    ----------
    ref : np.ndarray
        the reference image
    test : np.ndarray
        the tested image
    mask : np.ndarray, optional
        the mask for the ROI

    Notes
    -----
    Compute the metric only on magnetude.

    1/N * |ref - test|_2

    Returns
    -------
    mse: float, the mse

    """

    test, ref, mask = _preprocess_input(test, ref, mask)

    if mask is not None:
        test = mask * test
        ref = mask * ref

    return np.mean(np.square(test - ref))


def nrmse(test, ref, mask=None):
    """Return NRMSE

    Parameters
    ----------
    ref : np.ndarray
        the reference image
    test : np.ndarray
        the tested image
    mask : np.ndarray, optional
        the mask for the ROI

    Notes
    -----
    Compute the metric only on magnitude.

    Returns
    -------
    nrmse: float, the nrmse

    """

    test, ref, mask = _preprocess_input(test, ref, mask)

    if mask is not None:
        test = mask * test
        ref = mask * ref

    num = np.sqrt(mse(test, ref))
    deno = np.sqrt(np.mean((np.square(test))))

    return num / deno
