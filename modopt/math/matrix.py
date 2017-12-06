"""MATRIX ROUTINES

This module contains methods for matrix operations.

:Author: Samuel Farrens <samuel.farrens@gmail.com>

:Version: 1.2

:Date: 20/10/2017

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

    TODO
    ----
    Add citation and equation

    """

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
    else:
        return u, e


def nuclear_norm(data):
    """Nuclear norm

    This method computes the nuclear (or trace) norm of the input data.

    Parameters
    ----------
    data : np.ndarray
        Input data array

    Returns
    -------
    float nuclear norm value

    TODO
    ----
    Add equation

    """

    # Get SVD of the data.
    u, s, v = np.linalg.svd(data)

    # Return nuclear norm.
    return np.sum(s)


def project(u, v):
    """Project vector

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

    TODO
    ----
    Add equation

    """

    return np.inner(v, u) / np.inner(u, u) * u


def rot_matrix(angle):
    """Rotation matrix

    This method produces a 2x2 rotation matrix for the given input angle.

    Parameters
    ----------
    angle : float
        Rotation angle

    Returns
    -------
    np.ndarray 2x2 rotation matrix

    """

    return np.around(np.array([[np.cos(angle), -np.sin(angle)],
                     [np.sin(angle), np.cos(angle)]], dtype='float'), 10)


def rotate(matrix, angle):
    """Rotate

    This method rotates an input matrix about the input angle.

    Parameters
    ----------
    matrix : np.ndarray
        Input matrix array
    angle : float
        Rotation angle

    Returns
    -------
    np.ndarray rotated matrix

    Raises
    ------
    ValueError
        For invalid matrix shape

    """

    shape = np.array(matrix.shape)

    if shape[0] != shape[1]:
        raise ValueError('Input matrix must be square.')

    shift = (np.array(shape) - 1) // 2

    index = np.array(list(product(*np.array([np.arange(val) for val in
                     shape])))) - shift

    new_index = np.array(np.dot(index, rot_matrix(angle)), dtype='int') + shift
    new_index[new_index >= shape[0]] -= shape[0]

    return matrix[list(zip(new_index.T))].reshape(shape.T)


class PowerMethod(object):
    """Power method class

    This method performs implements power method to calculate the spectral
    radius of the input data

    Parameters
    ----------
    operator : class
        Operator class instance
    data_shape : tuple
        Shape of the data array
    auto_run : bool
        Option to automatically calcualte the spectral radius upon
        initialisation

    """

    def __init__(self, operator, data_shape, auto_run=True):

        self.op = operator
        self.data_shape = data_shape
        if auto_run:
            self.get_spec_rad()

    def set_initial_x(self):
        """Set initial value of x

        This method sets the initial value of x to an arrray of random values

        """

        return np.random.random(self.data_shape)

    def get_spec_rad(self, tolerance=1e-6, max_iter=10):
        """Get spectral radius

        This method calculates the spectral radius

        Parameters
        ----------
        tolerance : float, optional
            Tolerance threshold for convergence (default is "1e-6")
        max_iter : int, optional
            Maximum number of iterations

        """

        # Set (or reset) values of x.
        x_old = self.set_initial_x()

        # Iterate until the L2 norm of x converges.
        for i in range(max_iter):

            x_new = self.op(x_old) / np.linalg.norm(x_old)

            if(np.abs(np.linalg.norm(x_new) - np.linalg.norm(x_old)) <
               tolerance):
                print(' - Power Method converged after %d iterations!' %
                      (i + 1))
                break

            elif i == max_iter - 1:
                print(' - Power Method did not converge after %d '
                      'iterations!' % max_iter)

            np.copyto(x_old, x_new)

        self.spec_rad = np.linalg.norm(x_new)
        self.inv_spec_rad = 1.0 / self.spec_rad
