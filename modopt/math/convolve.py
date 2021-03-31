# -*- coding: utf-8 -*-

"""CONVOLUTION ROUTINES.

This module contains methods for convolution.

:Author: Samuel Farrens <samuel.farrens@cea.fr>

"""

import numpy as np
import scipy.signal

from modopt.base.np_adjust import rotate_stack
from modopt.interface.errors import warn

try:
    from astropy.convolution import convolve_fft
except ImportError:  # pragma: no cover
    import_astropy = False
    warn('astropy not found, will default to scipy for convolution')
else:
    import_astropy = True
try:
    import pyfftw
except ImportError:  # pragma: no cover
    pass
else:  # pragma: no cover
    scipy.fftpack = pyfftw.interfaces.scipy_fftpack
    warn('Using pyFFTW "monkey patch" for scipy.fftpack')


def convolve(input_data, kernel, method='scipy'):
    """Convolve data with kernel.

    This method convolves the input data with a given kernel using FFT and
    is the default convolution used for all routines

    Parameters
    ----------
    input_data : numpy.ndarray
        Input data array, normally a 2D image
    kernel : numpy.ndarray
        Input kernel array, normally a 2D kernel
    method : {'scipy', 'astropy'}, optional
        Convolution method (default is 'scipy')

    Returns
    -------
    numpy.ndarray
        Convolved data

    Raises
    ------
    ValueError
        If `data` and `kernel` do not have the same number of dimensions
    ValueError
        If `method` is not 'astropy' or 'scipy'

    Examples
    --------
    >>> import numpy as np
    >>> from modopt.math.convolve import convolve
    >>> a = np.arange(9).reshape(3, 3)
    >>> b = a + 10
    >>> convolve(a, b)
    array([[ 86., 170., 146.],
           [246., 444., 354.],
           [290., 494., 374.]])

    >>> convolve(a, b, method='astropy')
    array([[534., 525., 534.],
           [453., 444., 453.],
           [534., 525., 534.]])

    See Also
    --------
    scipy.signal.fftconvolve : scipy FFT convolution
    astropy.convolution.convolve_fft : astropy FFT convolution

    """
    if input_data.ndim != kernel.ndim:
        raise ValueError('Data and kernel must have the same dimensions.')

    if method not in {'astropy', 'scipy'}:
        raise ValueError('Invalid method. Options are "astropy" or "scipy".')

    if not import_astropy:  # pragma: no cover
        method = 'scipy'

    if method == 'astropy':
        return convolve_fft(
            input_data,
            kernel,
            boundary='wrap',
            crop=False,
            nan_treatment='fill',
            normalize_kernel=False,
        )

    elif method == 'scipy':
        return scipy.signal.fftconvolve(input_data, kernel, mode='same')


def convolve_stack(input_data, kernel, rot_kernel=False, method='scipy'):
    """Convolve stack of data with stack of kernels.

    This method convolves the input data with a given kernel using FFT and
    is the default convolution used for all routines

    Parameters
    ----------
    input_data : numpy.ndarray
        Input data array, normally a 2D image
    kernel : numpy.ndarray
        Input kernel array, normally a 2D kernel
    rot_kernel : bool
        Option to rotate kernels by 180 degrees (default is ``False``)
    method : {'astropy', 'scipy'}, optional
        Convolution method (default is 'scipy')

    Returns
    -------
    numpy.ndarray
        Convolved data

    Examples
    --------
    >>> import numpy as np
    >>> from modopt.math.convolve import convolve_stack
    >>> a = np.arange(18).reshape(2, 3, 3)
    >>> b = a + 10
    >>> convolve_stack(a, b, method='astropy')
    array([[[ 534.,  525.,  534.],
            [ 453.,  444.,  453.],
            [ 534.,  525.,  534.]],
    <BLANKLINE>
           [[2721., 2712., 2721.],
            [2640., 2631., 2640.],
            [2721., 2712., 2721.]]])

    >>> convolve_stack(a, b, method='astropy', rot_kernel=True)
    array([[[ 474.,  483.,  474.],
            [ 555.,  564.,  555.],
            [ 474.,  483.,  474.]],
    <BLANKLINE>
           [[2661., 2670., 2661.],
            [2742., 2751., 2742.],
            [2661., 2670., 2661.]]])

    See Also
    --------
    convolve : The convolution function called by convolve_stack

    """
    if rot_kernel:
        kernel = rotate_stack(kernel)

    return np.array([
        convolve(data_i, kernel_i, method=method)
        for data_i, kernel_i in zip(input_data, kernel)
    ])
