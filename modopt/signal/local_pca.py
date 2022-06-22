"""Local PCA filtering functions.

This module provides local low rank denoising methods.
The main entry point is ``local_svd_patch``, where several denoising methods
are available. This method works on patches of a N-D array, where the N-1
first dimensions represent space, and the last one some dynamic variation
(e.g time). Spatial patches are extracted and process using the available
methods.
"""

from types import MappingProxyType

import numpy as np
from scipy.linalg import svd, eigh, svdvals

def _patch_locs(v_shape, p_shape, p_ovl):
    """
    Get all the patch top-left corner locations.

    Parameters
    ----------
    vol_shape : tuple
        The volume shape
    patch_shape : tuple
        The patch shape
    patch_overlap : tuple
        The overlap of patch for each dimension.

    Returns
    -------
    numpy.ndarray
        All the patch top-left corner locations.
    """
    # Create an iterator for all the possible patches top-left corner location.
    if len(v_shape) != len(p_shape) or len(v_shape) != len(p_ovl):
        raise ValueError('Dimension mismatch between the arguments.')

    ranges = []
    for v_s, p_s, p_o in zip(v_shape, p_shape, p_ovl):
        if p_o >= p_s:
            raise ValueError(
                'Overlap should be a non-negative integer'
                + 'smaller than patch_size',
            )
        last_idx = v_s - p_s
        range_ = np.arange(0, last_idx, p_s - p_o, dtype=np.int32)
        if range_[-1] < last_idx:
            range_ = np.append(range_, last_idx)
        ranges.append(range_)
    # fast ND-Cartesian product from https://stackoverflow.com/a/11146645
    patch_locs = np.empty(
        [len(arr) for arr in ranges] + [len(p_shape)],
        dtype=np.int32,
    )
    for idx, coords in enumerate(np.ix_(*ranges)):
        patch_locs[..., idx] = coords

    return patch_locs.reshape(-1, len(p_shape))


def _patch_svd_analysis(input_data):
    """Return the centered SVD decomposition  X = U @ (S * Vt) + M.

    Parameters
    ----------
    input_data : numpy.ndarray
        The patch

    Returns
    -------
    u_vec, s_vals, v_vec, mean
    """
    mean = np.mean(input_data, axis=0)
    input_data -= mean
    # TODO  benchmark svd vs svds and order of data.
    u_vec, s_vals, v_vec = svd(input_data, full_matrices=False)

    return u_vec, s_vals, v_vec, mean


def _patch_svd_synthesis(u_vec, s_vals, v_vec, mean, idx):
    """
    Reconstruct X= (U @ (S * V)) + M with only the max_idx greatest component.

    U, S, V must be sorted in decreasing order.

    Parameters
    ----------
    u_vec : numpy.ndarray
    s_vals : numpy.ndarray
    v_vec : numpy.ndarray
    mean : numpy.ndarray
    idx : int

    Returns
    -------
    np.ndarray: The reconstructed matrix.
    """
    return (u_vec[:, :idx] @ (s_vals[:idx, None] * v_vec[:idx, :])) + mean


def _patch_eig_analysis(input_data, max_eig_val=10):
    """
    Return the eigen values and vectors of the autocorrelation of the patch.

    This method is surprisingly faster than the svd, but the eigen values
    are in increasing order.

    Parameters
    ----------
    input_data : np.ndarray
        A 2D Array
    max_eig_val : int, optional
       For faster results, only the ``max_eig_val`` biggest eigenvalues are
       computed. default = 10

    Returns
    -------
    A : numpy.ndarray
        The centered patch A = X - M
    d : numpy.ndarray
        The eigenvalues of A^H A
    W : numpy.ndarray
        The eigenvector matrix of A^H A
    M : numpy.ndarray
        The mean of the patch along the time axis
    """
    mean = np.mean(input_data, axis=0)
    data_centered = (input_data - mean)
    eig_vals, eig_vec = eigh(
        data_centered.conj().T @ data_centered,
        turbo=True,
        subset_by_index=(len(mean) - max_eig_val, len(mean) - 1),
    )

    return data_centered, eig_vals, eig_vec, mean


def _patch_eig_synthesis(data_centered, eig_vec, mean, max_val):
    """Reconstruction the denoise patch with truncated eigen decomposition.

    This implements equations (1) and (2) of :cite:`manjon2013`
    """
    eig_vec[:, :-max_val] = 0
    return ((data_centered @ eig_vec) @ eig_vec.conj().T) + mean


def patch_denoise_hybrid(patch, varest=0, **kwargs):
    """Denoise a patch using the Hybrid PCA method.

    Parameters
    ----------
    patch : numpy.ndarray
        The patch to process
    varest: float
        the noise variance a priori estimate for the patch.

    Returns
    -------
    patch_new : numpy.ndarray
        The processed patch
    noise_map : numpy.ndarray
        An estimation of the noise
    """
    p_center, eig_vals, eig_vec, p_tmean = _patch_eig_analysis(patch)
    n_sval_max = len(eig_vec)
    eig_vals /= n_sval_max
    maxidx = 0
    var = np.mean(eig_vals)
    while var > varest and maxidx < len(eig_vals)-2:
        maxidx += 1
        var = np.mean(eig_vals[:-maxidx])
    if maxidx == 0: # all eigen values are noise
        patch_new = np.zeros_like(patch) + p_tmean
    else:
        patch_new = _patch_eig_synthesis(p_center, eig_vec, p_tmean, maxidx)
    # Equation (3) of Manjon2013
    theta = 1.0 / (1.0 + maxidx)
    noise_map = varest * theta
    patch_new *= theta
    weights = theta

    return patch_new, noise_map, weights


def patch_denoise_mppca(patch, threshold_scale=1.0, **kwargs):
    r"""Denoise a patch using MP-PCA thresholding.

    Parameters
    ----------
    patch : np.ndarray
        the patch to process in a 2D form
    threshold_scale: float, default 1.0
        The estimated threshold will be multiplied by threshold scale.

    Returns
    -------
    patch_new: np.ndarray
        The weighted denoised patch
    noise_map: np.floating
        Estimation of the noise variance on the patch
    weights: np.floating
        The patch associated weights.

    Notes
    -----
    The patches weight are computed using Equation (3) of :cite:`manjon2013`

    .. math:: \theta = \frac{1}{1+\|\hat{\sigma_i}\|_0}

    ie, a patch with be more important if it has lesser singular values
    (:math:`\hat{sigma_i}`) after the thresholding.
    """
    p_center, eig_vals, eig_vec, p_tmean = _patch_eig_analysis(patch)
    n_voxels = len(p_center)
    n_sval_max = len(eig_vec)
    eig_vals /= n_sval_max
    maxidx = 0
    meanvar = np.mean(eig_vals) * (4 * np.sqrt((len(eig_vals) - maxidx + 1)/n_voxels))
    while meanvar < eig_vals[~maxidx] - eig_vals[0]:
        maxidx += 1
        meanvar = np.mean(eig_vals[:-maxidx]) * (4 * np.sqrt((n_sval_max - maxidx + 1)/n_voxels))
    var = np.mean(eig_vals[:len(eig_vals)-maxidx])

    thresh = var * threshold_scale ** 2

    maxidx = np.sum(eig_vals > thresh)

    if maxidx == 0:
        patch_new = np.zeros_like(patch) + p_tmean
    else:
        patch_new = _patch_eig_synthesis(p_center, eig_vec, p_tmean, maxidx)

    theta = 1.0 / (1.0 +  maxidx)
    noise_map = var * theta
    patch_new *= theta
    weights = theta

    return patch_new, noise_map, weights


def patch_denoise_raw(patch, threshold=0.0, **kwargs):
    """
    Denoise a patch using the singular value thresholding.

    Parameters
    ----------
    patch : numpy.ndarray
        The patch to process
    threshold_value : float
        The thresholding value for the patch

    Returns
    -------
    patch_new : numpy.ndarray
        The processed patch.
    weights : numpy.ndarray
        The weight associated with the patch.
    """
    # Centering for better precision in SVD
    u_vec, s_values, v_vec, p_tmean = _patch_svd_analysis(patch)

    maxidx = np.sum(s_values > threshold)
    if maxidx == 0:
        p_new = np.zeros_like(patch) + p_tmean
    else:
        s_values[s_values < threshold] = 0
        p_new = _patch_svd_synthesis(u_vec, s_values, v_vec, p_tmean, maxidx)

    # Equation 3 in Manjon 2013
    theta = 1.0 / (1.0 + maxidx)
    p_new *= theta
    weights = theta

    return p_new, weights, np.NaN


# initialisation dispatch function for the different methods.

def _init_svd_thresh_raw(*args, **denoiser_kwargs):
    """Initialisation for raw thresholding method."""
    if 'threshold' not in denoiser_kwargs:
        e_s = 'For RAW denoiser, the threshold must be provided as a named argument.'
        raise ValueError(e_s)
    return patch_denoise_raw,  denoiser_kwargs


def _init_svd_thresh_nordic(patch_shape=None, data_shape=None, noise_std=None, denoiser_kwargs=None, **kwargs):
    """Initialisation for NORDIC Thresholding method."""
    # NORDIC DENOISER
    # The threshold is the same for all patches and estimated from
    # Monte-Carlo simulations, and scaled using the noise_level.
    num_iters = denoiser_kwargs.get("threshold_estimation_iters", 10)
    max_sval = 0
    for _ in range(num_iters):
        max_sval += max(svd(
            np.random.randn(np.prod(patch_shape), data_shape[-1]),
            compute_uv=False))
    max_sval /= num_iters

    if isinstance(noise_std, (float, np.floating)):
        noise_std =  noise_std
    elif isinstance(noise_std, np.ndarray):
        noise_std = np.mean(noise_std)
    else:
        e_s = 'For NORDIC the noise level must be either an'
        e_s += 'array or a float specifying the std in the volume.'
        raise ValueError(e_s)

    scale_factor = denoiser_kwargs.get('threshold_scale_factor', 1.0)
    denoiser_kwargs['threshold'] = max_sval * noise_std * scale_factor

    return patch_denoise_raw, denoiser_kwargs


def _init_svd_thresh_mppca(patch_shape=None, data_shape=None, **denoiser_kwargs):
    """Initialisation for the MP-PCA denoiser."""
    if "threshold_scale" not in denoiser_kwargs:
        denoiser_kwargs["threshold_scale"] = 1 + np.sqrt(data_shape[-1]/np.prod(patch_shape))

    return patch_denoise_mppca, denoiser_kwargs


def _init_svd_thresh_hybrid(noise_std=None, denoiser_kwargs=None, **kwargs):
    if not isinstance(noise_std, (float, np.floating, np.ndarray)):
        e_s = 'For HYBRID the noise level must be either an '
        e_s += 'array or a float specifying the std.'
        raise ValueError(e_s)
    return patch_denoise_hybrid, denoiser_kwargs


def _init_svd_thresh_donoho(*args, denoiser_kwargs=None, **kwargs):
    pass


_INIT_SVD_THRESH = MappingProxyType({
    'RAW': _init_svd_thresh_raw,
    'NORDIC': _init_svd_thresh_nordic,
    'MP-PCA': _init_svd_thresh_mppca,
    'HYBRID': _init_svd_thresh_hybrid,
    'DONOHO': _init_svd_thresh_donoho,
})


def local_svd_thresh(
    input_data,
    patch_shape,
    patch_overlap,
    mask=None,
    noise_std=None,
    threshold_method='MPPCA',
    use_center_of_patch=False,
    **denoiser_kwargs,
):
    r"""
    Perform local low rank denoising.

    This method perform a denoising operation by processing spatial patches of
    dynamic data (ie. an array of with N first spatial dimension and a last
    temporal one).
    Each patch is denoised by thresholding its singular values decomposition,
    and recombined using a weighted average.

    Parameters
    ----------
    input_data : numpy.ndarray
        The input data to denoise of N dimensions. Spatial dimension are the
        first N-1 ones, on which patch will be extracted. The last dimension
        corresponds to dynamic evolution (e.g. time).
    patch_shape: tuple
        The shape of the local patch.
    patch_overlap : tuple
        A tuple specifying the amount of pixel/voxel overlapping in each
        dimension.
    threshold_method : callable or str, optional.
        One of the supported noise thresholding method. default "RAW".
    noise_std: float or numpy.ndarray, default None.
        If float, the noise standard deviation.
        If array, A noise-only volume, that will be used to estimate the noise
        level globaly or locally (depending on the threshold method).
        If None (default) it will be estimated with the corresponding method if
        threshold_method is "MPPCA" or "HYBRID".
        A value must be specified for "NORDIC". If is an array, then the
        average over the patch will be considered.
    use_center_of_patch: bool, default False
        If the overlap is maximum (ie patch is a sliding window), and if the
        patch shape in each dimension is odd,
        if ``use_center_of_patch`` is True, then only the center value of the
        patch is use for the output.
    denoiser_kwargs: dict
        Extra argument to pass to the patch denoising function.

    Returns
    -------
    output_data: numpy.ndarray
        The denoised data.
    patch_weight: numpy.ndarray
        The accumulated weight of each patch for each pixel.
    noise_std_map: numpy.ndarray
        The estimated spatial map of the temporal noise std.

    Notes
    -----
    The implemented threshold methods are:
     * "RAW"
       The singular value are hard-thresholded by noise_level.

     * "MP-PCA"
       The noise level :math:`\hat\sigma` is determined using
       Marschenko-Pastur's Law. Then, the threshold is perform for 
       :math:`\hat\tau = (\tau \hat\sigma)^2` where :math:`\tau=1+\sqrt{M/N}`
       where M and N are the number of colons and row of the Casorati Matrix of
       the extracted patch.

     * "NORDIC"
       The noise level :math:`\sigma` estimation must be provided.
       The threshold value will be determining by taking the average of the
       maximum singular value of 10 MxN  random matrices with noise level
       :math:`\sigma`.
       The threshold can further be tweak by providing threshold_scale_factor
       (default is ``1.0``, ie no scaling)

     * "HYBRID"
       The noise level :math:`\sigma` estimation must be provided.
       The number of lowest singular values c to remove is such that
       :math:`\sum_i^c{\lambda_i}/c\le \sigma`

    Related Implementations can be found in [1]_, [2]_, and [3]_

    References
    ----------
    .. [1] https://github.com/dipy/dipy/blob/master/dipy/denoise/localpca.py
    .. [2] https://github.com/SteenMoeller/NORDIC_Raw
    .. [3] https://github.com/RafaelNH/Hybrid-PCA/
    """
    data_shape = input_data.shape
    if np.prod(patch_shape) < data_shape[-1]:
        raise ValueError('the number of voxel in patch is smaller than the \
        last dimension, this makes an ill-conditioned matrix for SVD.')

    output_data = np.zeros_like(input_data)
    if denoiser_kwargs is None:
        denoiser_kwargs = {}

    use_center_of_patch &= all(ps % 2 == 1  and ps - po == 1
                               for ps, po in zip(patch_shape, patch_overlap))
    if use_center_of_patch:
        patch_center = tuple(slice(ps // 2, ps//2+1)  for ps in patch_shape)
    patchs_weight = np.zeros(data_shape[:-1], np.float32)
    noise_std_estimate = np.zeros(data_shape[:-1], dtype=np.float32)

    thresh_func, denoiser_kwargs = INIT_SVD_THRESH[threshold_method](
        patch_shape=patch_shape,
        data_shape=data_shape,
        noise_std=noise_std,
        denoiser_kwargs=denoiser_kwargs,
    )
    if threshold_method == "HYBRID":
        if isinstance(noise_std, (float, np.floating)):
            noise_var = noise_std ** 2 * np.ones_like(noise_std_estimate)
        else:
            noise_var = noise_std ** 2
    # main loop
    # TODO Paralellization over patches
    for patch_tl in _patch_locs(data_shape[:-1], patch_shape, patch_overlap):
        # building patch_slice
        # a (N-1)D slice for the input data
        # and extracting one patch for processing.
        patch_slice = (slice(patch_tl[0], patch_tl[0] + patch_shape[0]),)
        for tl, shape in zip(patch_tl[1:], patch_shape[1:]):
            patch_slice = (*patch_slice, slice(tl, tl + shape))
        if mask is not None and not np.any(mask[patch_slice]):
            continue  # patch is outside the mask.
        # building the casoratti matrix
        patch = np.reshape(
            input_data[patch_slice],
            (-1, input_data.shape[-1]),
        )
        if threshold_method == "HYBRID":
             denoiser_kwargs['varest'] = np.mean(noise_var[patch_slice] ** 2)
        p_denoise, *extras = thresh_func(
            patch,
            **denoiser_kwargs,
        )


        p_denoise = np.reshape(p_denoise, (*patch_shape, -1))
        if use_center_of_patch:
            patch_center_img = tuple(slice(ptl+ps//2, ptl+ps//2 + 1) for ptl, ps in zip (patch_tl, patch_shape))
            output_data[patch_center_img] = p_denoise[patch_center]
        else:
            output_data[patch_slice] += p_denoise
            if len(extras) > 1:
                patchs_weight[patch_slice] += extras[0]
                noise_std_estimate[patch_slice] += extras[1]
            else:
                patchs_weight[patch_slice] += extras[0]
    if not use_center_of_patch:
        # Averaging the overlapping pixels.
        output_data /= patchs_weight[..., None]
        noise_std_estimate /= patchs_weight

    if mask is not None:
        output_data[~mask] = 0

    return output_data, patchs_weight, noise_std_estimate
