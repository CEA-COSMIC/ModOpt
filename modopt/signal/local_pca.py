"""Local PCA filtering functions."""

import numpy as np
from scipy.linalg import svd


def _get_patch_locs(vol_shape, patch_shape, patch_overlap):
    """Get all the patch top-left corner locations.

    Parameters
    ----------
    vol_shape: tuple
    patch_shape: tuple
    patch_overlap: tuple

    Returns
    -------
    numpy.ndarray
        All the patch top-left corner locations.
    """
    # Create an iterator for all the possible patches top-left corner location.

    if not (len(vol_shape) == len(patch_shape) == len(patch_overlap)):
        raise ValueError("Dimension mismatch between the arguments.")

    ranges = []
    for v_shape, p_shape, p_ovl in zip(vol_shape, patch_shape, patch_overlap):
        last_idx = v_shape - p_shape
        range_ = np.arange(
            0,
            last_idx,
            p_shape - p_ovl,
            dtype=np.int32,
        )
        if not range_ or range_[-1] < last_idx:
            range_ = np.append(range_, last_idx)
        ranges.append(range_)
    # fast ND-Cartesian product from https://stackoverflow.com/a/11146645
    patch_locs = np.empty(
        [len(arr) for arr in ranges] + [len(patch_shape)],
        dtype=np.int32,
    )
    for idx, coords in enumerate(np.ix_(*ranges)):
        patch_locs[..., idx] = coords

    patch_locs = patch_locs.reshape(-1, len(patch_shape))
    return patch_locs

def _get_svd_thresh_mppca(input_data, nvoxels):
    """Estimate the threshold using Marshenko-Pastur's Law.

    Parameters
    ----------
    input_data : array like
        1D array of singular values.
    nvoxels: int
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


def local_svd_thresh(
    input_data,
    patch_shape,
    patch_overlap,
    mask=None,
    threshold='MPPCA',
    extra_factor=None,
):
    """
    Perform local low rank denoising.

    This method perform a denoising operation by processing spatial patches of
    dynamic data (ie. an array of with N first spatial dimension and a last
    temporal one).
    Each patch is denoised by thresholding its singular values decomposition,
    and recombined using a weighted average.

    Parameters
    ----------
    input_data: numpy.ndarray
        The input data to denoise of N dimensions. Spatial dimension are the first
        N-1 ones, on which patch will be extracted. The last dimension corresponds
        to dynamic evolution (e.g. time).
    patch_shape: tuple
        The shape of the local patch.
    patch_overlap: tuple
        A tuple specifying the amount of pixel/voxel overlapping in each
        dimension.
    threshold_method: {"RAW", "HYBRID", "MPPCA", "NORDIC"}
        One of the supported noise thresholding method. default "RAW".
    noise_level: float or numpy.ndarray, default None.
        The noise level value, as a scalar if homogeneous, as an array if not.
        If None (default) it will be estimated with the corresponding method if
        threshold_method is "MPPCA" or "HYBRID".
        A value must be specified for "NORDIC". If is an array, then the average
        over the patch will be considered.

    extra_factor: float, default None
        Extra factor for the threshold.
        If None, it will be set using random matrix theory.

    Returns
    -------
    output_data: numpy.ndarray
        The denoised data.
    noise_level: numpy.ndarray
        The estimated spatial map of the noise level.

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
     * "NORDIC":
       The noise level :math:`\sigma` estimation must be provided. the threshold
       value will be determining by taking the average of the maximum singular value
       of 10 MxN  random matrices with noise level :math:`\sigma` . 
     * "HYBRID"
       The noise level :math:`\sigma` estimation must be provided. the number of
       lowest singular values c to remove is such that :math:`\sum_i^c{\lambda_i}/c
       \le \sigma` 



    Related Implementations can be found in [1]_, [2]_, and [3]_

    References
    ----------
    .. [1] https://github.com/dipy/dipy/blob/master/dipy/denoise/localpca.py
    .. [2] https://github.com/SteenMoeller/NORDIC_Raw
    .. [3] https://github.com/RafaelNH/Hybrid-PCA/
"""
    data_shape = input_data.shape

    # Using random matrix theory for estimating the extra factor.
    if extra_factor is None:
        extra_factor = 1 + np.sqrt(data_shape[-1]/np.product(patch_shape))

    output_data = np.zeros_like(input_data)
    noise_map = np.zeros(data_shape[:-1], dtype=np.float32)
    mask = mask or np.ones(data_shape[:-1], dtype=bool)

    if np.sum(patch_overlap) > 0:
        patchs_weight = np.zeros(data_shape[:-1], np.float32)
    # main loop
    # TODO Paralellization over patches
    for patch_tl in _get_patch_locs(data_shape, patch_shape, patch_overlap):
        # building patch_slice
        # a (N-1)D slice for the input data
        # and extracting one patch for processing.
        patch_slice = np.s_[patch_tl[0]:patch_tl[0] + patch_shape[0]]
        for tl, shape in zip(patch_tl[1:], patch_shape[1:]):
            patch_slice = np.s_[patch_slice, tl:shape + tl]

        if not np.any(mask[patch_slice]):
            continue  # patch is outside the mask.
        # building the casoratti matrix
        patch = np.reshape(
            input_data[patch_slice, :],
            (-1, input_data.shape[-1]),
        )
        patch_tmean = np.mean(patch, axis=0)

        # Centering for better precision in SVD
        patch -= patch_tmean
        u_vec, s_values, v_vec = svd(patch, full_matrices=False)
        # S value are ascending (:croissant:).
        norm_sval = s_values ** 2 / patch.shape[0]

        # Custom thresholding functions.
        if threshold == 'MP-PCA':
            patch_var, n_vals = _get_svd_thresh_mppca(norm_sval, patch.size)
            threshold = patch_var * (extra_factor ** 2)
        if threshold == 'HYBRID-PCA':
            patch_var = np.std(patch) ** 2

        s_values[norm_sval < threshold] = 0
        n_vals = np.sum(s_values > 0)
        patch = u_vec @ (s_values[n_vals:] * v_vec[n_vals:, :]) + patch_tmean
        patch = np.reshape(patch, (*patch_shape, input_data.shape[-1]))

        # This is equation 3 in Manjon 2013:
        patch_theta = 1.0 / (1.0 + data_shape[-1] - n_vals)

        output_data[patch_slice, :] += patch * patch_theta
        if np.sum(patch_overlap) > 0:
            patchs_weight[patch_slice] += patch_theta
            noise_map[patch_slice] = patch_var * patch_theta
    # Averaging the overlapping pixels.
    if patch_overlap > 0:
        output_data /= patchs_weight[..., None]
        noise_map /= patchs_weight[..., None]

    output_data[~mask] = 0

    return output_data, noise_map
