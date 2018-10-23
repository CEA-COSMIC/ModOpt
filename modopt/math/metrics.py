"""METRICS

This module contains classes of different metric functions for optimization.

:Author: Benoir Sarthou

"""

import numpy as np
from scipy.ndimage import uniform_filter, gaussian_filter
from numpy.lib.arraypad import _validate_lengths

dtype_range = {np.bool_: (False, True),
               np.bool8: (False, True),
               np.uint8: (0, 255),
               np.uint16: (0, 65535),
               np.uint32: (0, 2**32 - 1),
               np.uint64: (0, 2**64 - 1),
               np.int8: (-128, 127),
               np.int16: (-32768, 32767),
               np.int32: (-2**31, 2**31 - 1),
               np.int64: (-2**63, 2**63 - 1),
               np.float16: (-1, 1),
               np.float32: (-1, 1),
               np.float64: (-1, 1)}


def crop(ar, crop_width, copy=False, order='K'):
    """Crop

    Crop array `ar` by `crop_width` along each dimension.

    Parameters
    ----------
    ar : array-like of rank N
        Input array.
    crop_width : {sequence, int}
        Number of values to remove from the edges of each axis.
        ``((before_1, after_1),`` ... ``(before_N, after_N))`` specifies
        unique crop widths at the start and end of each axis.
        ``((before, after),)`` specifies a fixed start and end crop
        for every axis.
        ``(n,)`` or ``n`` for integer ``n`` is a shortcut for
        before = after = ``n`` for all axes.
    copy : bool, optional
        If `True`, ensure the returned array is a contiguous copy. Normally,
        a crop operation will return a discontiguous view of the underlying
        input array.
    order : {'C', 'F', 'A', 'K'}, optional
        If ``copy==True``, control the memory layout of the copy. See
        ``np.copy``.

    Returns
    -------
    cropped : array
        The cropped array. If ``copy=False`` (default), this is a sliced
        view of the input array.

    """

    ar = np.array(ar, copy=False)
    crops = _validate_lengths(ar, crop_width)
    slices = [slice(a, ar.shape[i] - b) for i, (a, b) in enumerate(crops)]
    if copy:
        cropped = np.array(ar[slices], order=order, copy=True)
    else:
        cropped = ar[slices]
    return cropped


def _compare_ssim(X, Y, win_size=None, gradient=False,
                  data_range=None, multichannel=False, gaussian_weights=False,
                  full=False, **kwargs):
    """Compute the mean structural similarity index between two images.

    Parameters
    ----------
    X, Y : ndarray
        Image.  Any dimensionality.
    win_size : int or None
        The side-length of the sliding window used in comparison.  Must be an
        odd value.  If `gaussian_weights` is True, this is ignored and the
        window size will depend on `sigma`.
    gradient : bool, optional
        If True, also return the gradient with respect to Y.
    data_range : float, optional
        The data range of the input image (distance between minimum and
        maximum possible values).  By default, this is estimated from the image
        data-type.
    multichannel : bool, optional
        If True, treat the last dimension of the array as channels. Similarity
        calculations are done independently for each channel then averaged.
    gaussian_weights : bool, optional
        If True, each patch has its mean and variance spatially weighted by a
        normalized Gaussian kernel of width sigma=1.5.
    full : bool, optional
        If True, return the full structural similarity image instead of the
        mean value.

    Other Parameters
    ----------------
    use_sample_covariance : bool
        if True, normalize covariances by N-1 rather than, N where N is the
        number of pixels within the sliding window.
    K1 : float
        algorithm parameter, K1 (small constant, see [1]_)
    K2 : float
        algorithm parameter, K2 (small constant, see [1]_)
    sigma : float
        sigma for the Gaussian when `gaussian_weights` is True.

    Returns
    -------
    mssim : float
        The mean structural similarity over the image.
    grad : ndarray
        The gradient of the structural similarity index between X and Y [2]_.
        This is only returned if `gradient` is set to True.
    S : ndarray
        The full SSIM image.  This is only returned if `full` is set to True.

    Notes
    -----
    To match the implementation of Wang et. al. [1]_, set `gaussian_weights`
    to True, `sigma` to 1.5, and `use_sample_covariance` to False.

    References
    ----------
    .. [1] Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P.
       (2004). Image quality assessment: From error visibility to
       structural similarity. IEEE Transactions on Image Processing,
       13, 600-612.
       https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf,
       DOI:10.1109/TIP.2003.819861
    .. [2] Avanaki, A. N. (2009). Exact global histogram specification
       optimized for structural similarity. Optical Review, 16, 613-621.
       http://arxiv.org/abs/0901.0065,
       DOI:10.1007/s10043-009-0119-z

    """

    if not X.dtype == Y.dtype:
        raise ValueError('Input images must have the same dtype.')

    if not X.shape == Y.shape:
        raise ValueError('Input images must have the same dimensions.')

    if multichannel:
        # loop over channels
        args = dict(win_size=win_size,
                    gradient=gradient,
                    data_range=data_range,
                    multichannel=False,
                    gaussian_weights=gaussian_weights,
                    full=full)
        args.update(kwargs)
        nch = X.shape[-1]
        mssim = np.empty(nch)
        if gradient:
            G = np.empty(X.shape)
        if full:
            S = np.empty(X.shape)
        for ch in range(nch):
            ch_result = compare_ssim(X[..., ch], Y[..., ch], **args)
            if gradient and full:
                mssim[..., ch], G[..., ch], S[..., ch] = ch_result
            elif gradient:
                mssim[..., ch], G[..., ch] = ch_result
            elif full:
                mssim[..., ch], S[..., ch] = ch_result
            else:
                mssim[..., ch] = ch_result
        mssim = mssim.mean()
        if gradient and full:
            return mssim, G, S
        elif gradient:
            return mssim, G
        elif full:
            return mssim, S
        else:
            return mssim

    K1 = kwargs.pop('K1', 0.01)
    K2 = kwargs.pop('K2', 0.03)
    sigma = kwargs.pop('sigma', 1.5)
    if K1 < 0:
        raise ValueError("K1 must be positive")
    if K2 < 0:
        raise ValueError("K2 must be positive")
    if sigma < 0:
        raise ValueError("sigma must be positive")
    use_sample_covariance = kwargs.pop('use_sample_covariance', True)

    if win_size is None:
        if gaussian_weights:
            win_size = 11  # 11 to match Wang et. al. 2004
        else:
            win_size = 7   # backwards compatibility

    if np.any((np.asarray(X.shape) - win_size) < 0):
        raise ValueError(
            "win_size exceeds image extent.  If the input is a multichannel "
            "(color) image, set multichannel=True.")

    if not (win_size % 2 == 1):
        raise ValueError('Window size must be odd.')

    if data_range is None:
        dmin, dmax = dtype_range[X.dtype.type]
        data_range = dmax - dmin

    ndim = X.ndim

    if gaussian_weights:
        # sigma = 1.5 to approximately match filter in Wang et. al. 2004
        # this ends up giving a 13-tap rather than 11-tap Gaussian
        filter_func = gaussian_filter
        filter_args = {'sigma': sigma}

    else:
        filter_func = uniform_filter
        filter_args = {'size': win_size}

    # ndimage filters need floating point data
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)

    NP = win_size ** ndim

    # filter has already normalized by NP
    if use_sample_covariance:
        cov_norm = NP / (NP - 1)  # sample covariance
    else:
        cov_norm = 1.0  # population covariance to match Wang et. al. 2004

    # compute (weighted) means
    ux = filter_func(X, **filter_args)
    uy = filter_func(Y, **filter_args)

    # compute (weighted) variances and covariances
    uxx = filter_func(X * X, **filter_args)
    uyy = filter_func(Y * Y, **filter_args)
    uxy = filter_func(X * Y, **filter_args)
    vx = cov_norm * (uxx - ux * ux)
    vy = cov_norm * (uyy - uy * uy)
    vxy = cov_norm * (uxy - ux * uy)

    R = data_range
    C1 = (K1 * R) ** 2
    C2 = (K2 * R) ** 2

    A1, A2, B1, B2 = ((2 * ux * uy + C1,
                       2 * vxy + C2,
                       ux ** 2 + uy ** 2 + C1,
                       vx + vy + C2))
    D = B1 * B2
    S = (A1 * A2) / D

    # to avoid edge effects will ignore filter radius strip around edges
    pad = (win_size - 1) // 2

    # compute (weighted) mean of ssim
    mssim = crop(S, pad).mean()

    if gradient:
        # The following is Eqs. 7-8 of Avanaki 2009.
        grad = filter_func(A1 / D, **filter_args) * X
        grad += filter_func(-S / B2, **filter_args) * Y
        grad += filter_func((ux * (A2 - A1) - uy * (B2 - B1) * S) / D,
                            **filter_args)
        grad *= (2 / X.size)

        if full:
            return mssim, grad, S
        else:
            return mssim, grad
    else:
        if full:
            return mssim, S
        else:
            return mssim


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

    test, ref, mask = _preprocess_input(test, ref, mask)
    assim, ssim = _compare_ssim(test, ref, full=True)

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
