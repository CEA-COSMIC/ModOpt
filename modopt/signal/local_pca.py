"""Local PCA filtering functions."""

import numpy as np
from scipy.linalg import svd

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

    for p_s, p_o in zip(p_shape, p_ovl):
        if p_o >= p_s:
            e_s = 'Overlap should be a non-negative integer, smaller than patch_size'
            raise ValueError(e_s)
    ranges = []
    for v_s, p_s, p_o in zip(v_shape, p_shape, p_ovl):
        last_idx = v_s - p_s
        range_ = np.arange(0, last_idx, p_s - p_o, dtype=np.int32,)
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


def _get_svd_thresh_mppca(input_data, nvoxels):
    """
    Estimate the threshold using Marshenko-Pastur's Law.

    Parameters
    ----------
    input_data : array like
        1D array of singular values.
    nvoxels : int
        The total number of voxels used to computes the singular values.

    Returns
    -------
    var : float
        Estimation of the noise variance
    ncomps : int
        Number of eigenvalues related to noise

    Notes
    -----
    This is based on the algorithm described in :cite:`veraart2016`.
    Similar implementation can be found in the dipy [1] package


    References
    ----------
    .. [1] https://github.com/dipy/dipy/blob/master/dipy/denoise/localpca.py`
    """
    sigma2 = np.mean(input_data)
    n_vals = input_data.size - 1
    band = input_data[n_vals] - input_data[0]
    band -= 4 * np.sqrt((n_vals + 1.0) / nvoxels) * sigma2
    while band > 0:
        sigma2 = np.mean(input_data[:n_vals])
        n_vals = n_vals - 1
        band = input_data[n_vals] - input_data[0]
        band -= 4 * np.sqrt((n_vals + 1.0) / nvoxels) * sigma2
        ncomps = n_vals + 1
    return sigma2, ncomps


def patch_denoise_hybrid(patch, noise_level=1.0,  **kwargs):
    """Denoise a patch using the Hybrid PCA method."""
    patch_tmean = np.mean(patch, axis=0)

    # Centering for better precision in SVD
    patch -= patch_tmean
    u_vec, s_values, v_vec = svd(patch, full_matrices=False)
    norm_sval = s_values ** 2 / patch.shape[0]
    # get an average noise std.
    varest = np.mean(noise_level)
    # get the maximal index of thresholded values.
    norm_sval_var = np.mean(norm_sval)
    maxidx = norm_sval.size - 1
    while norm_sval_var > varest:
        norm_sval_var = np.mean(norm_sval[maxidx + 1:])
        maxidx -= 1
    maxidx +=1

    patch_new = u_vec[:, maxidx:] @ s_values[maxidx:] * v_vec[maxidx:]
    theta = 1.0 / (1.0 + patch.shape[1] - maxidx)
    noise_map = varest * theta
    patch_new *= theta
    weights = theta

    return patch_new, noise_map, weights

DENOISE_METHOD ={
    'MP-PCA': patch_denoise_mppca,
    'HYBRID': patch_denoise_hybrid,
    'RAW': patch_denoise_raw,
    'NORDIC': patch_denoise_nordic,

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

INIT_SVD_THRESH = {
    "RAW": _init_svd_thresh_raw,
    "NORDIC": _init_svd_thresh_nordic,
    "MP-PCA": _init_svd_thresh_mppca,
    "HYBRID": _init_svd_thresh_hybrid,
    "DONOHO": _init_svd_thresh_donoho,
}

def local_svd_thresh(
    input_data,
    patch_shape,
    patch_overlap,
    mask=None,
    noise_std=None,
    threshold_method='MPPCA',
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
            input_data[(*patch_slice, None)],
            (-1, input_data.shape[-1]),
        )
        p_denoise, p_noise_map, p_weight = DENOISE_METHOD[threshold](
            patch,
            noise_level=noise_level,
            threshold_value=threshold_value,
        )


        p_denoise = np.reshape(p_denoise, (*patch_shape, -1))
        output_data[patch_slice] += p_denoise
        if len(extras) > 1:
            patchs_weight[patch_slice] += extras[0]
            noise_std_estimate[patch_slice] += extras[1]
        else:
            patchs_weight[patch_slice] += extras[0]

    # Averaging the overlapping pixels.
    output_data /= patchs_weight[..., None]
    noise_std_estimate /= patchs_weight

    if mask is not None:
        output_data[~mask] = 0

    return output_data, noise_map
