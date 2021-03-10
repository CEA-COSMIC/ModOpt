# -*- coding: utf-8 -*-

"""STATISTICS ROUTINES.

This module contains methods for basic statistics.

:Author: Samuel Farrens <samuel.farrens@cea.fr>

"""

import numpy as np

try:
    from astropy.convolution import Gaussian2DKernel
except ImportError:  # pragma: no cover
    import_astropy = False
else:
    import_astropy = True


def gaussian_kernel(data_shape, sigma, norm='max'):
    """Gaussian kernel.

    This method produces a Gaussian kerenal of a specified size and dispersion

    Parameters
    ----------
    data_shape : tuple
        Desiered shape of the kernel
    sigma : float
        Standard deviation of the kernel
    norm : {'max', 'sum', 'none'}, optional
        Normalisation of the kerenl (options are 'max', 'sum' or 'none',
        default is 'max')

    Returns
    -------
    numpy.ndarray
        Kernel

    Raises
    ------
    ImportError
        If Astropy package not found
    ValueError
        For invalid norm

    Examples
    --------
    >>> from modopt.math.stats import gaussian_kernel
    >>> gaussian_kernel((3, 3), 1)
    array([[0.36787944, 0.60653066, 0.36787944],
           [0.60653066, 1.        , 0.60653066],
           [0.36787944, 0.60653066, 0.36787944]])

    >>> gaussian_kernel((3, 3), 1, norm='sum')
    array([[0.07511361, 0.1238414 , 0.07511361],
           [0.1238414 , 0.20417996, 0.1238414 ],
           [0.07511361, 0.1238414 , 0.07511361]])

    """
    if not import_astropy:  # pragma: no cover
        raise ImportError('Astropy package not found.')

    if norm not in {'max', 'sum', 'none'}:
        raise ValueError('Invalid norm, options are "max", "sum" or "none".')

    kernel = np.array(
        Gaussian2DKernel(sigma, x_size=data_shape[1], y_size=data_shape[0]),
    )

    if norm == 'max':
        return kernel / np.max(kernel)

    elif norm == 'sum':
        return kernel / np.sum(kernel)

    elif norm == 'none':
        return kernel


def mad(input_data):
    r"""Median absolute deviation.

    This method calculates the median absolute deviation of the input data.

    Parameters
    ----------
    input_data : numpy.ndarray
        Input data array

    Returns
    -------
    float
        MAD value

    Examples
    --------
    >>> import numpy as np
    >>> from modopt.math.stats import mad
    >>> a = np.arange(9).reshape(3, 3)
    >>> mad(a)
    2.0

    Notes
    -----
    The MAD is calculated as follows:

    .. math::

        \mathrm{MAD} = \mathrm{median}\left(|X_i - \mathrm{median}(X)|\right)

    See Also
    --------
    numpy.median : median function used

    """
    return np.median(np.abs(input_data - np.median(input_data)))


def mse(data1, data2):
    """Mean Squared Error.

    This method returns the Mean Squared Error (MSE) between two data sets.

    Parameters
    ----------
    data1 : numpy.ndarray
        First data set
    data2 : numpy.ndarray
        Second data set

    Returns
    -------
    float
        Mean squared error

    Examples
    --------
    >>> import numpy as np
    >>> from modopt.math.stats import mse
    >>> a = np.arange(9).reshape(3, 3)
    >>> mse(a, a + 2)
    4.0

    """
    return np.mean((data1 - data2) ** 2)


def psnr(data1, data2, method='starck', max_pix=255):
    r"""Peak Signal-to-Noise Ratio.

    This method calculates the Peak Signal-to-Noise Ratio between an two data
    sets

    Parameters
    ----------
    data1 : numpy.ndarray
        First data set
    data2 : numpy.ndarray
        Second data set
    method : {'starck', 'wiki'}, optional
        PSNR implementation (default  is 'starck')
    max_pix : int, optional
        Maximum number of pixels (default is ``255``)

    Returns
    -------
    float
        PSNR value

    Raises
    ------
    ValueError
        For invalid PSNR method

    Examples
    --------
    >>> import numpy as np
    >>> from modopt.math.stats import psnr
    >>> a = np.arange(9).reshape(3, 3)
    >>> psnr(a, a + 2)
    12.041199826559248

    >>> psnr(a, a + 2, method='wiki')
    42.11020369539948

    Notes
    -----
    'starck':

        Implements eq.3.7 from :cite:`starck2010`

    'wiki':

        Implements PSNR equation on
        https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

        .. math::

            \mathrm{PSNR} = 20\log_{10}(\mathrm{MAX}_I -
            10\log_{10}(\mathrm{MSE}))

    """
    if method == 'starck':
        return (
            20 * np.log10(
                (data1.shape[0] * np.abs(np.max(data1) - np.min(data1)))
                / np.linalg.norm(data1 - data2),
            )
        )

    elif method == 'wiki':
        return (20 * np.log10(max_pix) - 10 * np.log10(mse(data1, data2)))

    raise ValueError(
        'Invalid PSNR method. Options are "starck" and "wiki"',
    )


def psnr_stack(data1, data2, metric=np.mean, method='starck'):
    """Peak Signa-to-Noise for stack of images.

    This method calculates the PSNRs for two stacks of 2D arrays.
    By default the metod returns the mean value of the PSNRs, but any other
    metric can be used.

    Parameters
    ----------
    data1 : numpy.ndarray
        Stack of images, 3D array
    data2 : numpy.ndarray
        Stack of recovered images, 3D array
    metric : function
        The desired metric to be applied to the PSNR values (default is
        ``numpy.mean``)
    method : {'starck', 'wiki'}, optional
        PSNR implementation (default is 'starck')

    Returns
    -------
    float
        Metric result of PSNR values

    Raises
    ------
    ValueError
        For invalid input data dimensions

    Examples
    --------
    >>> import numpy as np
    >>> from modopt.math.stats import psnr_stack
    >>> a = np.arange(18).reshape(2, 3, 3)
    >>> psnr_stack(a, a + 2)
    12.041199826559248

    See Also
    --------
    numpy.mean : default metric

    """
    if data1.ndim != 3 or data2.ndim != 3:
        raise ValueError('Input data must be a 3D np.ndarray')

    return metric([
        psnr(i_elem, j_elem, method=method)
        for i_elem, j_elem in zip(data1, data2)
    ])


def sigma_mad(input_data):
    r"""MAD Standard Deviation.

    This method calculates the standard deviation of the input data from the
    MAD.

    Parameters
    ----------
    input_data : numpy.ndarray
        Input data array

    Returns
    -------
    float
        Sigma value

    Examples
    --------
    >>> import numpy as np
    >>> from modopt.math.stats import sigma_mad
    >>> a = np.arange(9).reshape(3, 3)
    >>> sigma_mad(a)
    2.9652

    Notes
    -----
    This function can be used for estimating the standeviation of the noise in
    imgaes.

    Sigma is calculated as follows:

    .. math::

        \sigma = 1.4826 \mathrm{MAD}(X)

    """
    return 1.4826 * mad(input_data)
