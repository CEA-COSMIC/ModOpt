# -*- coding: utf-8 -*-

"""MATRIX ROUTINES

This module contains methods for matrix operations.

:Author: Samuel Farrens <samuel.farrens@cea.fr>

"""

from __future__ import division
from builtins import range, zip
import numpy as np
from itertools import product


def gram_schmidt(matrix, return_opt='orthonormal'):
    r"""Gram-Schmit

    This method orthonormalizes the row vectors of the input matrix.

    Parameters
    ----------
    matrix : np.ndarray
        Input matrix array
    return_opt : str {orthonormal, orthogonal, both}
        Option to return u, e or both.

    Returns
    -------
    Lists of orthogonal vectors, u, and/or orthonormal vectors, e

    Examples
    --------
    >>> from modopt.math.matrix import gram_schmidt
    >>> a = np.arange(9).reshape(3, 3)
    >>> gram_schmidt(a)
    array([[ 0.        ,  0.4472136 ,  0.89442719],
           [ 0.91287093,  0.36514837, -0.18257419],
           [-1.        ,  0.        ,  0.        ]])

    Notes
    -----
    Implementation from:
    https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process

    """

    if return_opt not in ('orthonormal', 'orthogonal', 'both'):
        raise ValueError('Invalid return_opt, options are: "orthonormal", '
                         '"orthogonal" or "both"')

    u = []
    e = []

    for vector in matrix:

        if len(u) == 0:
            u_now = vector
        else:
            u_now = vector - sum([project(u_i, vector) for u_i in u])

        u.append(u_now)
        e.append(u_now / np.linalg.norm(u_now, 2))

    u = np.array(u)
    e = np.array(e)

    if return_opt == 'orthonormal':
        return e
    elif return_opt == 'orthogonal':
        return u
    elif return_opt == 'both':
        return u, e


def nuclear_norm(data):
    r"""Nuclear norm

    This method computes the nuclear (or trace) norm of the input data.

    Parameters
    ----------
    data : np.ndarray
        Input data array

    Returns
    -------
    float nuclear norm value

    Examples
    --------
    >>> from modopt.math.matrix import nuclear_norm
    >>> a = np.arange(9).reshape(3, 3)
    >>> nuclear_norm(a)
    15.49193338482967

    Notes
    -----
    Implements the following equation:

    .. math::
        \|\mathbf{A}\|_* = \sum_{i=1}^{\min\{m,n\}} \sigma_i (\mathbf{A})

    """

    # Get SVD of the data.
    u, s, v = np.linalg.svd(data)

    # Return nuclear norm.
    return np.sum(s)


def project(u, v):
    r"""Project vector

    This method projects vector v onto vector u.

    Parameters
    ----------
    u : np.ndarray
        Input vector
    v : np.ndarray
        Input vector

    Returns
    -------
    np.ndarray projection

    Examples
    --------
    >>> from modopt.math.matrix import project
    >>> a = np.arange(3)
    >>> b = a + 3
    >>> project(a, b)
    array([ 0. ,  2.8,  5.6])

    Notes
    -----
    Implements the following equation:

    .. math::
        \textrm{proj}_\mathbf{u}(\mathbf{v}) = \frac{\langle\mathbf{u},
        \mathbf{v}\rangle}{\langle\mathbf{u}, \mathbf{u}\rangle}\mathbf{u}

    (see https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process)

    """

    return np.inner(v, u) / np.inner(u, u) * u


def rot_matrix(angle):
    r"""Rotation matrix

    This method produces a 2x2 rotation matrix for the given input angle.

    Parameters
    ----------
    angle : float
        Rotation angle in radians

    Returns
    -------
    np.ndarray 2x2 rotation matrix

    Examples
    --------
    >>> from modopt.math.matrix import rot_matrix
    >>> rot_matrix(np.pi / 6)
    array([[ 0.8660254, -0.5      ],
           [ 0.5      ,  0.8660254]])

    Notes
    -----
    Implements the following equation:

    .. math::
        R(\theta) = \begin{bmatrix}
            \cos(\theta) & -\sin(\theta) \\
            \sin(\theta) & \cos(\theta)
        \end{bmatrix}

    """

    return np.around(np.array([[np.cos(angle), -np.sin(angle)],
                     [np.sin(angle), np.cos(angle)]], dtype='float'), 10)


def rotate(matrix, angle):
    r"""Rotate

    This method rotates an input matrix about the input angle.

    Parameters
    ----------
    matrix : np.ndarray
        Input matrix array
    angle : float
        Rotation angle in radians

    Returns
    -------
    np.ndarray rotated matrix

    Raises
    ------
    ValueError
        For invalid matrix shape

    Examples
    --------
    >>> from modopt.math.matrix import rotate
    >>> a = np.arange(9).reshape(3, 3)
    >>> rotate(a, np.pi / 2)
    array([[2, 5, 8],
           [1, 4, 7],
           [0, 3, 6]])

    """

    shape = np.array(matrix.shape)

    if shape[0] != shape[1]:
        raise ValueError('Input matrix must be square.')

    shift = (shape - 1) // 2

    index = np.array(list(product(*np.array([np.arange(val) for val in
                     shape])))) - shift

    new_index = np.array(np.dot(index, rot_matrix(angle)), dtype='int') + shift
    new_index[new_index >= shape[0]] -= shape[0]

    return matrix[tuple(zip(new_index.T))].reshape(shape.T)


class PowerMethod(object):
    """Power method class

    This method performs implements power method to calculate the spectral
    radius of the input data

    Parameters
    ----------
    operator : function
        Operator function
    data_shape : tuple
        Shape of the data array
    data_type : type {float, complex}, optional
        Random data type (default is float)
    auto_run : bool, optional
        Option to automatically calcualte the spectral radius upon
        initialisation (default is True)
    verbose : bool, optional
        Optional verbosity (default is False)

    Examples
    --------
    >>> from modopt.math.matrix import PowerMethod
    >>> np.random.seed(1)
    >>> pm = PowerMethod(lambda x: x.dot(x.T), (3, 3))
     - Power Method converged after 4 iterations!
    >>> pm.spec_rad
    0.90429242629600848
    >>> pm.inv_spec_rad
    1.1058369736612865

    Notes
    -----
    Implementation from: https://en.wikipedia.org/wiki/Power_iteration

    """

    def __init__(self, operator, data_shape, data_type=float, auto_run=True,
                 verbose=False):

        self._operator = operator
        self._data_shape = data_shape
        self._data_type = data_type
        self._verbose = verbose
        if auto_run:
            self.get_spec_rad()

    def _set_initial_x(self):
        """Set initial value of x

        This method sets the initial value of x to an arrray of random values

        Returns
        -------
        np.ndarray of random values of the same shape as the input data

        """

        return np.random.random(self._data_shape).astype(self._data_type)

    def get_spec_rad(self, tolerance=1e-6, max_iter=20, extra_factor=1.0):
        """Get spectral radius

        This method calculates the spectral radius

        Parameters
        ----------
        tolerance : float, optional
            Tolerance threshold for convergence (default is "1e-6")
        max_iter : int, optional
            Maximum number of iterations (default is 20)
        extra_factor : float, optional
            Extra multiplicative factor for calculating the spectral radius
            (default is 1.0)

        """

        # Set (or reset) values of x.
        x_old = self._set_initial_x()

        # Iterate until the L2 norm of x converges.
        for i in range(max_iter):

            x_old_norm = np.linalg.norm(x_old)

            x_new = self._operator(x_old) / x_old_norm

            x_new_norm = np.linalg.norm(x_new)

            if(np.abs(x_new_norm - x_old_norm) < tolerance):
                if self._verbose:
                    print(' - Power Method converged after %d iterations!' %
                          (i + 1))
                break

            elif i == max_iter - 1 and self._verbose:
                print(' - Power Method did not converge after %d '
                      'iterations!' % max_iter)

            np.copyto(x_old, x_new)

        self.spec_rad = x_new_norm * extra_factor
        self.inv_spec_rad = 1.0 / self.spec_rad
