# -*- coding: utf-8 -*-

"""MATRIX ROUTINES.

This module contains methods for matrix operations.

:Author: Samuel Farrens <samuel.farrens@cea.fr>

"""

from itertools import product

import numpy as np

from modopt.base.backend import get_array_module, get_backend


def gram_schmidt(matrix, return_opt='orthonormal'):
    """Gram-Schmit.

    This method orthonormalizes the row vectors of the input matrix.

    Parameters
    ----------
    matrix : numpy.ndarray
        Input matrix array
    return_opt : {'orthonormal', 'orthogonal', 'both'}
        Option to return u, e or both, (default is 'orthonormal')

    Returns
    -------
    tuple or numpy.ndarray
        Orthogonal vectors, u, and/or orthonormal vectors, e

    Raises
    ------
    ValueError
        For invalid return option

    Examples
    --------
    >>> import numpy as np
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
    if return_opt not in {'orthonormal', 'orthogonal', 'both'}:
        raise ValueError(
            'Invalid return_opt, options are: "orthonormal", "orthogonal" or '
            + '"both"',
        )

    u_vec = []
    e_vec = []

    for vector in matrix:

        if u_vec:
            u_now = vector - sum(project(u_i, vector) for u_i in u_vec)
        else:
            u_now = vector

        u_vec.append(u_now)
        e_vec.append(u_now / np.linalg.norm(u_now, 2))

    u_vec = np.array(u_vec)
    e_vec = np.array(e_vec)

    if return_opt == 'orthonormal':
        return e_vec
    elif return_opt == 'orthogonal':
        return u_vec
    elif return_opt == 'both':
        return u_vec, e_vec


def nuclear_norm(input_data):
    r"""Nuclear norm.

    This method computes the nuclear (or trace) norm of the input data.

    Parameters
    ----------
    input_data : numpy.ndarray
        Input data array

    Returns
    -------
    float
        Nuclear norm value

    Examples
    --------
    >>> import numpy as np
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
    _, singular_values, _ = np.linalg.svd(input_data)

    # Return nuclear norm.
    return np.sum(singular_values)


def project(u_vec, v_vec):
    r"""Project vector.

    This method projects vector v onto vector u.

    Parameters
    ----------
    u_vec : numpy.ndarray
        Input vector
    v_vec : numpy.ndarray
        Input vector

    Returns
    -------
    numpy.ndarray
        Projection

    Examples
    --------
    >>> import numpy as np
    >>> from modopt.math.matrix import project
    >>> a = np.arange(3)
    >>> b = a + 3
    >>> project(a, b)
    array([0. , 2.8, 5.6])

    Notes
    -----
    Implements the following equation:

    .. math::
        \textrm{proj}_\mathbf{u}(\mathbf{v}) = \frac{\langle\mathbf{u},
        \mathbf{v}\rangle}{\langle\mathbf{u}, \mathbf{u}\rangle}\mathbf{u}

    (see https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process)

    """
    return np.inner(v_vec, u_vec) / np.inner(u_vec, u_vec) * u_vec


def rot_matrix(angle):
    r"""Rotation matrix.

    This method produces a 2x2 rotation matrix for the given input angle.

    Parameters
    ----------
    angle : float
        Rotation angle in radians

    Returns
    -------
    numpy.ndarray
        2x2 rotation matrix

    Examples
    --------
    >>> import numpy as np
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
    return np.around(
        np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
            dtype='float',
        ),
        10,
    )


def rotate(matrix, angle):
    """Rotate.

    This method rotates an input matrix about the input angle.

    Parameters
    ----------
    matrix : numpy.ndarray
        Input matrix array
    angle : float
        Rotation angle in radians

    Returns
    -------
    numpy.ndarray
        Rotated matrix

    Raises
    ------
    ValueError
        For invalid matrix shape

    Examples
    --------
    >>> import numpy as np
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

    index = (
        np.array(list(product(*np.array([np.arange(sval) for sval in shape]))))
        - shift
    )

    new_index = np.array(np.dot(index, rot_matrix(angle)), dtype='int') + shift
    new_index[new_index >= shape[0]] -= shape[0]

    return matrix[tuple(zip(new_index.T))].reshape(shape.T)


class PowerMethod(object):
    """Power method class.

    This method performs implements power method to calculate the spectral
    radius of the input data

    Parameters
    ----------
    operator : function
        Operator function
    data_shape : tuple
        Shape of the data array
    data_type : {``float``, ``complex``}, optional
        Random data type (default is ``float``)
    auto_run : bool, optional
        Option to automatically calcualte the spectral radius upon
        initialisation (default is ``True``)
    verbose : bool, optional
        Optional verbosity (default is ``False``)

    Examples
    --------
    >>> import numpy as np
    >>> from modopt.math.matrix import PowerMethod
    >>> np.random.seed(1)
    >>> pm = PowerMethod(lambda x: x.dot(x.T), (3, 3))
    >>> np.around(pm.spec_rad, 6)
    0.904292
    >>> np.around(pm.inv_spec_rad, 6)
    1.105837

    Notes
    -----
    Implementation from: https://en.wikipedia.org/wiki/Power_iteration

    """

    def __init__(
        self,
        operator,
        data_shape,
        data_type=float,
        auto_run=True,
        compute_backend='numpy',
        verbose=False,
    ):

        self._operator = operator
        self._data_shape = data_shape
        self._data_type = data_type
        self._verbose = verbose
        xp, compute_backend = get_backend(compute_backend)
        self.xp = xp
        self.compute_backend = compute_backend
        if auto_run:
            self.get_spec_rad()

    def _set_initial_x(self):
        """Set initial value of x.

        This method sets the initial value of x to an arrray of random values

        Returns
        -------
        numpy.ndarray
            Random values of the same shape as the input data

        """
        return self.xp.random.random(self._data_shape).astype(self._data_type)

    def get_spec_rad(self, tolerance=1e-6, max_iter=20, extra_factor=1.0):
        """Get spectral radius.

        This method calculates the spectral radius

        Parameters
        ----------
        tolerance : float, optional
            Tolerance threshold for convergence (default is ``1e-6``)
        max_iter : int, optional
            Maximum number of iterations (default is ``20``)
        extra_factor : float, optional
            Extra multiplicative factor for calculating the spectral radius
            (default is ``1.0``)

        """
        # Set (or reset) values of x.
        x_old = self._set_initial_x()

        # Iterate until the L2 norm of x converges.
        for i_elem in range(max_iter):

            xp = get_array_module(x_old)

            x_old_norm = xp.linalg.norm(x_old)

            x_new = self._operator(x_old) / x_old_norm

            x_new_norm = xp.linalg.norm(x_new)

            if (xp.abs(x_new_norm - x_old_norm) < tolerance):
                message = (
                    ' - Power Method converged after {0} iterations!'
                )
                if self._verbose:
                    print(message.format(i_elem + 1))
                break

            elif i_elem == max_iter - 1 and self._verbose:
                message = (
                    ' - Power Method did not converge after {0} iterations!'
                )
                print(message.format(max_iter))

            xp.copyto(x_old, x_new)

        self.spec_rad = x_new_norm * extra_factor
        self.inv_spec_rad = 1.0 / self.spec_rad
