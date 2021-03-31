# -*- coding: utf-8 -*-

"""NOISE ROUTINES.

This module contains methods for adding and removing noise from data.

:Author: Samuel Farrens <samuel.farrens@cea.fr>

"""

from builtins import zip

import numpy as np

from modopt.base.backend import get_array_module


def add_noise(input_data, sigma=1.0, noise_type='gauss'):
    """Add noise to data.

    This method adds Gaussian or Poisson noise to the input data

    Parameters
    ----------
    input_data : numpy.ndarray, list or tuple
        Input data array
    sigma : float or list, optional
        Standard deviation of the noise to be added ('gauss' only, default is
        ``1.0``)
    noise_type : {'gauss', 'poisson'}
        Type of noise to be added (default is 'gauss')

    Returns
    -------
    numpy.ndarray
        Input data with added noise

    Raises
    ------
    ValueError
        If `noise_type` is not 'gauss' or 'poisson'
    ValueError
        If number of `sigma` values does not match the first dimension of the
        input data

    Examples
    --------
    >>> import numpy as np
    >>> from modopt.signal.noise import add_noise
    >>> x = np.arange(9).reshape(3, 3).astype(float)
    >>> x
    array([[0., 1., 2.],
           [3., 4., 5.],
           [6., 7., 8.]])
    >>> np.random.seed(1)
    >>> add_noise(x, noise_type='poisson')
    array([[ 0.,  2.,  2.],
           [ 4.,  5., 10.],
           [11., 15., 18.]])

    >>> import numpy as np
    >>> from modopt.signal.noise import add_noise
    >>> x = np.zeros(5)
    >>> x
    array([0., 0., 0., 0., 0.])
    >>> np.random.seed(1)
    >>> add_noise(x, sigma=2.0)
    array([ 3.24869073, -1.22351283, -1.0563435 , -2.14593724,  1.73081526])

    """
    input_data = np.array(input_data)

    if noise_type not in {'gauss', 'poisson'}:
        raise ValueError(
            'Invalid noise type. Options are "gauss" or "poisson"',
        )

    if isinstance(sigma, (list, tuple, np.ndarray)):
        if len(sigma) != input_data.shape[0]:
            raise ValueError(
                'Number of sigma values must match first dimension of input '
                + 'data',
            )

    if noise_type == 'gauss':
        random = np.random.randn(*input_data.shape)

    elif noise_type == 'poisson':
        random = np.random.poisson(np.abs(input_data))

    if isinstance(sigma, (int, float)):
        return input_data + sigma * random

    noise = np.array([sig * rand for sig, rand in zip(sigma, random)])

    return input_data + noise


def thresh(input_data, threshold, threshold_type='hard'):
    r"""Threshold data.

    This method perfoms hard or soft thresholding on the input data

    Parameters
    ----------
    input_data : numpy.ndarray, list or tuple
        Input data array
    threshold : float or numpy.ndarray
        Threshold level(s)
    threshold_type : {'hard', 'soft'}
        Type of noise to be added (default is 'hard')

    Returns
    -------
    numpy.ndarray
        Thresholded data

    Raises
    ------
    ValueError
        If `threshold_type` is not 'hard' or 'soft'

    Notes
    -----
    Implements one of the following two equations:

    * Hard Threshold
        .. math::
            \mathrm{HT}_\lambda(x) =
            \begin{cases}
            x & \text{if } |x|\geq\lambda \\
            0 & \text{otherwise}
            \end{cases}

    * Soft Threshold
        .. math::
            \mathrm{ST}_\lambda(x) =
            \begin{cases}
            x-\lambda\text{sign}(x) & \text{if } |x|\geq\lambda \\
            0 & \text{otherwise}
            \end{cases}

    Examples
    --------
    >>> import numpy as np
    >>> from modopt.signal.noise import thresh
    >>> np.random.seed(1)
    >>> x = np.random.randint(-9, 9, 10)
    >>> x
    array([-4,  2,  3, -1,  0,  2, -4,  6, -9,  7])
    >>> thresh(x, 4)
    array([-4,  0,  0,  0,  0,  0, -4,  6, -9,  7])

    >>> import numpy as np
    >>> from modopt.signal.noise import thresh
    >>> np.random.seed(1)
    >>> x = np.random.ranf((3, 3))
    >>> x
    array([[4.17022005e-01, 7.20324493e-01, 1.14374817e-04],
           [3.02332573e-01, 1.46755891e-01, 9.23385948e-02],
           [1.86260211e-01, 3.45560727e-01, 3.96767474e-01]])
    >>> thresh(x, 0.2, threshold_type='soft')
    array([[0.217022  , 0.52032449, 0.        ],
           [0.10233257, 0.        , 0.        ],
           [0.        , 0.14556073, 0.19676747]])

    """
    xp = get_array_module(input_data)

    input_data = xp.array(input_data)

    if threshold_type not in {'hard', 'soft'}:
        raise ValueError(
            'Invalid threshold type. Options are "hard" or "soft"',
        )

    if threshold_type == 'soft':
        denominator = xp.maximum(xp.finfo(np.float64).eps, xp.abs(input_data))
        max_value = xp.maximum((1.0 - threshold / denominator), 0)

        return xp.around(max_value * input_data, decimals=15)

    return input_data * (xp.abs(input_data) >= threshold)
