# -*- coding: utf-8 -*-

"""METRICS.

This module contains classes of different metric functions for optimization.

:Author: Benoir Sarthou

"""

import numpy as np

from modopt.base.backend import move_to_cpu

try:
    from skimage.metrics import structural_similarity as compare_ssim
except ImportError:  # pragma: no cover
    import_skimage = False
else:
    import_skimage = True


def min_max_normalize(img):
    """Min-Max Normalize.

    Centre and normalize a given array.

    Parameters
    ----------
    img : numpy.ndarray
        Input image

    Returns
    -------
    numpy.ndarray
        Centred and normalized array

    """
    min_img = img.min()
    max_img = img.max()

    return (img - min_img) / (max_img - min_img)


def _preprocess_input(test, ref, mask=None):
    """Proprocess Input.

    Wrapper to the metric.

    Parameters
    ----------
    ref : numpy.ndarray
        The reference image
    test : numpy.ndarray
        The tested image
    mask : numpy.ndarray, optional
        The mask for the ROI (default is ``None``)

    Raises
    ------
    ValueError
        For invalid mask value

    Notes
    -----
    Compute the metric only on magnetude.

    Returns
    -------
    float
        The SNR

    """
    test = np.abs(np.copy(test)).astype('float64')
    ref = np.abs(np.copy(ref)).astype('float64')
    test = min_max_normalize(test)
    ref = min_max_normalize(ref)

    if (not isinstance(mask, np.ndarray)) and (mask is not None):
        message = (
            'Mask should be None, or a numpy.ndarray, got "{0}" instead.'
        )
        raise ValueError(message.format(mask))

    if mask is None:
        return test, ref, None

    return test, ref, mask


def ssim(test, ref, mask=None):
    """Structural Similarity (SSIM).

    Calculate the SSIM between a test image and a reference image.

    Parameters
    ----------
    ref : numpy.ndarray
        The reference image
    test : numpy.ndarray
        The tested image
    mask : numpy.ndarray, optional
        The mask for the ROI (default is ``None``)

    Raises
    ------
    ImportError
        If Scikit-Image package not found

    Notes
    -----
    Compute the metric only on magnetude.

    Returns
    -------
    float
        The SNR

    """
    if not import_skimage:  # pragma: no cover
        raise ImportError(
            'Required version of Scikit-Image package not found'
            + 'see documentation for details: https://cea-cosmic.'
            + 'github.io/ModOpt/#optional-packages',
        )

    test, ref, mask = _preprocess_input(test, ref, mask)
    test = move_to_cpu(test)
    assim, ssim_value = compare_ssim(test, ref, full=True)

    if mask is None:
        return assim

    return (mask * ssim_value).sum() / mask.sum()


def snr(test, ref, mask=None):
    """Signal-to-Noise Ratio (SNR).

    Calculate the SNR between a test image and a reference image.

    Parameters
    ----------
    ref: numpy.ndarray
        The reference image
    test: numpy.ndarray
        The tested image
    mask: numpy.ndarray, optional
        The mask for the ROI (default is ``None``)

    Notes
    -----
    Compute the metric only on magnetude.

    Returns
    -------
    float
        The SNR

    """
    test, ref, mask = _preprocess_input(test, ref, mask)

    if mask is not None:
        test = mask * test

    num = np.mean(np.square(test))
    deno = mse(test, ref)

    return 10.0 * np.log10(num / deno)


def psnr(test, ref, mask=None):
    """Peak Signal-to-Noise Ratio (PSNR).

    Calculate the PSNR between a test image and a reference image.

    Parameters
    ----------
    ref : numpy.ndarray
        The reference image
    test : numpy.ndarray
        The tested image
    mask : numpy.ndarray, optional
        The mask for the ROI (default is ``None``)

    Notes
    -----
    Compute the metric only on magnetude.

    Returns
    -------
    float
        The PSNR

    """
    test, ref, mask = _preprocess_input(test, ref, mask)

    if mask is not None:
        test = mask * test
        ref = mask * ref

    num = np.max(np.abs(test))
    deno = mse(test, ref)

    return 10.0 * np.log10(num / deno)


def mse(test, ref, mask=None):
    r"""Mean Squared Error (MSE).

    Calculate the MSE between a test image and a reference image.

    Parameters
    ----------
    ref : numpy.ndarray
        The reference image
    test : numpy.ndarray
        The tested image
    mask : numpy.ndarray, optional
        The mask for the ROI (default is ``None``)

    Notes
    -----
    Compute the metric only on magnetude.

    .. math::
        1/N * \|ref - test\|_2

    Returns
    -------
    float
        The MSE

    """
    test, ref, mask = _preprocess_input(test, ref, mask)

    if mask is not None:
        test = mask * test
        ref = mask * ref

    return np.mean(np.square(test - ref))


def nrmse(test, ref, mask=None):
    """Return NRMSE.

    Parameters
    ----------
    ref : numpy.ndarray
        The reference image
    test : numpy.ndarray
        The tested image
    mask : numpy.ndarray, optional
        The mask for the ROI (default is ``None``)

    Notes
    -----
    Compute the metric only on magnitude.

    Returns
    -------
    float
        The NRMSE

    """
    test, ref, mask = _preprocess_input(test, ref, mask)

    if mask is not None:
        test = mask * test
        ref = mask * ref

    num = np.sqrt(mse(test, ref))
    deno = np.sqrt(np.mean((np.square(test))))

    return num / deno
