# -*- coding: utf-8 -*-

"""DATA TRANSFORM ROUTINES

This module contains methods for transforming data.

:Author: Samuel Farrens <samuel.farrens@gmail.com>

:Version: 1.2

:Date: 20/10/2017

"""

from __future__ import division
from builtins import range
import numpy as np
from scipy.ndimage import gaussian_filter
from itertools import islice, product


def cube2map(data_cube, layout):
    """Cube to Map

    This method transforms the input data from a 3D cube to a 2D map with a
    specified layout

    Parameters
    ----------
    data_cube : np.ndarray
        Input data cube, 3D array of 2D images
    Layout : tuple
        2D layout of 2D images

    Returns
    -------
    np.ndarray 2D map

    Raises
    ------
    ValueError
        For invalid layout

    """

    if data_cube.shape[0] != np.prod(layout):
        raise ValueError('The desired layout must match the number of input '
                         'data layers.')

    return np.vstack([np.hstack(data_cube[slice(layout[1] * i, layout[1] *
                      (i + 1))]) for i in range(layout[0])])


def map2cube(data_map, layout):
    """Map to cube

    This method transforms the input data from a 2D map with given layout to
    a 3D cube

    Parameters
    ----------
    data_map : np.ndarray
        Input data map, 2D array
    layout : tuple
        2D layout of 2D images

    Returns
    -------
    np.ndarray 3D cube

    Raises
    ------
    ValueError
        For invalid layout

    """

    if np.all(np.array(data_map.shape) % np.array(layout)) != 0:
        raise ValueError('The desired layout must be a multiple of the number '
                         'pixels in the data map.')

    d_shape = np.array(data_map.shape) // np.array(layout)

    return np.array([data_map[(slice(i * d_shape[0], (i + 1) * d_shape[0]),
                    slice(j * d_shape[1], (j + 1) * d_shape[1]))] for i in
                    range(layout[0]) for j in range(layout[1])])


def map2matrix(data_map, layout):
    """Map to Matrix

    This method transforms a 2D map to a 2D matrix

    Parameters
    ----------
    data_map : np.ndarray
        Input data map, 2D array
    layout : tuple
        2D layout of 2D images

    Returns
    -------
    np.ndarray 2D matrix

    Raises
    ------
    ValueError
        For invalid layout

    """

    layout = np.array(layout)

    # Select n objects
    n_obj = np.prod(layout)

    # Get the shape of the galaxy images
    gal_shape = (np.array(data_map.shape) // layout)[0]

    # Stack objects from map
    data_matrix = []

    for i in range(n_obj):
        lower = (gal_shape * (i // layout[1]),
                 gal_shape * (i % layout[1]))
        upper = (gal_shape * (i // layout[1] + 1),
                 gal_shape * (i % layout[1] + 1))
        data_matrix.append((data_map[lower[0]:upper[0],
                            lower[1]:upper[1]]).reshape(gal_shape ** 2))

    return np.array(data_matrix).T


def matrix2map(data_matrix, map_shape):
    """Matrix to Map

    This method transforms a 2D matrix to a 2D map

    Parameters
    ----------
    data_matrix : np.ndarray
        Input data matrix, 2D array
    map_shape : tuple
        2D shape of the output map

    Returns
    -------
    np.ndarray 2D map

    Raises
    ------
    ValueError
        For invalid layout

    """

    map_shape = np.array(map_shape)

    # Get the shape and layout of the galaxy images
    gal_shape = np.sqrt(data_matrix.shape[0])
    layout = np.array(map_shape // np.repeat(gal_shape, 2), dtype='int')

    # Map objects from matrix
    data_map = np.zeros(map_shape)

    temp = data_matrix.reshape(gal_shape, gal_shape, data_matrix.shape[1])

    for i in range(data_matrix.shape[1]):
        lower = (gal_shape * (i // layout[1]),
                 gal_shape * (i % layout[1]))
        upper = (gal_shape * (i // layout[1] + 1),
                 gal_shape * (i % layout[1] + 1))
        data_map[lower[0]:upper[0], lower[1]:upper[1]] = temp[:, :, i]

    return data_map


def cube2matrix(data_cube):
    """Cube to Matrix

    This method transforms a 3D cube to a 2D matrix

    Parameters
    ----------
    data_cube : np.ndarray
        Input data cube, 3D array

    Returns
    -------
    np.ndarray 2D matrix

    """

    return data_cube.reshape([data_cube.shape[0]] +
                             [np.prod(data_cube.shape[1:])]).T


def matrix2cube(data_matrix, im_shape):
    """Matrix to Cube

    This method transforms a 2D matrix to a 3D cube

    Parameters
    ----------
    data_matrix : np.ndarray
        Input data cube, 2D array
    im_shape : tuple
        2D shape of the individual images

    Returns
    -------
    np.ndarray 3D cube

    """

    return data_matrix.T.reshape([data_matrix.shape[1]] + list(im_shape))
