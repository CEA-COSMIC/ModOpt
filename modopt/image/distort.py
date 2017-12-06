# -*- coding: utf-8 -*-

"""IMAGE DISTORTION ROUTINES

This module contains methods for playing around with image properties.

:Author: Samuel Farrens <samuel.farrens@gmail.com>

:Version: 1.1

:Date: 20/10/2017

"""

from __future__ import division
import numpy as np
from itertools import product
from modopt.base.np_adjust import pad2d


def downsample(image, factor):
    """Downsample

    This method downsamples (decimates) an image.

    Parameters
    ----------
    image : np.ndarray
        Input image array
    factor : int
        Downsampling factor

    Returns
    -------
    np.ndarray downsampled image array

    Raises
    ------
    ValueError
        For invalid downsampling factor

    """

    factor = np.array(factor)

    if not np.all(factor > 0):
        raise ValueError('The downsampling factor values must be > 0.')

    if factor.size == 1:
        return image[0::factor, 0::factor]

    elif factor.size == 2:
        return image[0::factor[0], 0::factor[1]]

    else:
        raise ValueError('The downsampling factor can only contain one or ' +
                         'two values.')


def resize_even_image(image):
    """Resize even image

    This method returns an image with odd dimensions.

    Parameters
    ----------
    image : np.ndarray
        Input image array

    Returns
    -------
    np.ndarray resized image array

    """

    return image[[slice(x) for x in (np.array(image.shape) +
                  np.array(image.shape) % 2 - 1)]]


def roll_2d(data, roll_rad=(1, 1)):
    """Roll in 2D

    This method rolls an array in 2 dimensions.

    Parameters
    ----------
    data : np.ndarray
        Input 2D data array
    roll_rad : tuple
        Roll radius in each dimension

    Returns
    -------
    np.ndarray rolled array

    """

    return np.roll(np.roll(data, roll_rad[1], axis=1), roll_rad[0], axis=0)


def rot_and_roll(data):
    """Rotate and roll

    This method rotates (by 180 deg) and rolls a 2D array.

    Parameters
    ----------
    data : np.ndarray
        Input 2D data array

    Returns
    -------
    np.ndarray rotated and rolled array

    """

    return roll_2d(np.rot90(data, 2), -(np.array(data.shape) // 2))


def gen_mask(kernel_shape, image_shape):
    """Generate mask

    This method generates an image mask.

    Parameters
    ----------
    kernel_shape : tuple
        Shape of kernel
    image_shape : tuple
        Shape of image

    Returns
    -------
    np.ndarray boolean mask

    """

    kernel_shape = np.array(kernel_shape)
    image_shape = np.array(image_shape)

    shape_diff = image_shape - kernel_shape

    mask = np.ones(image_shape, dtype=bool)

    if shape_diff[0] > 0:
        mask[-shape_diff[0]:] = False
    if shape_diff[1] > 0:
        mask[:, -shape_diff[1]:] = False

    return roll_2d(mask, -(kernel_shape // 2))


def roll_sequence(data_shape):
    """Roll sequence

    This method generates the roll sequence for a 2D array.

    Parameters
    ----------
    data_shape : tuple
        Shape of data

    Returns
    -------
    list of roll radii

    """

    data_shape = np.array(data_shape)

    return list(product(*(range(data_shape[0]), range(data_shape[1]))))


def kernel_pattern(kernel_shape, mask):
    """Kernel pattern

    This method generates the kernel pattern. Rather than padding the kernel
    with zeroes to match the image size one simply extracts the series of
    repitions of the base kernel patterns.

    Parameters
    ----------
    kernel_shape : tuple
        Shape of kernel
    mask : np.ndarray
        Boolean mask

    Returns
    -------
    np.ndarray kernel pattern

    """

    kernel_shape = np.array(kernel_shape)

    kernel_buffer = 1 - np.array(kernel_shape) % 2

    n_rep_axis1 = sum(1 - mask[:, 0])
    n_rep_axis2 = sum(1 - mask[0])

    if np.any(mask[:, 0] is False):
        pos_1 = np.where(mask[:, 0] is False)[0][0] - 1 + kernel_buffer[0]

    if np.any(mask[0] is False):
        pos_2 = np.where(mask[0] is False)[0][0] - 1 + kernel_buffer[1]

    pattern = np.arange(np.prod(kernel_shape)).reshape(kernel_shape)

    for i in range(n_rep_axis1):
        pattern = np.insert(pattern, pos_1, pattern[pos_1], axis=0)

    for i in range(n_rep_axis2):
        pattern = np.insert(pattern, pos_2, pattern[:, pos_2], axis=1)

    return pattern.reshape(pattern.size)


def rearrange_kernel(kernel, data_shape=None):
    """Rearrange kernel

    This method rearanges the input kernel elements for vector multiplication.
    The input kernel is padded with zeroes to match the image size.

    Parameters
    ----------
    kernel : np.ndarray
        Input kernel array
    data_shape : tuple
        Shape of the data

    Returns
    -------
    np.ndarray rearanged matrix of kernel elements

    """

    # Define kernel shape.
    kernel_shape = np.array(kernel.shape)

    # Set data shape if not provided.
    if isinstance(data_shape, type(None)):
        data_shape = kernel_shape
    else:
        data_shape = np.array(data_shape)

    # Set the length of the output matrix rows.
    vec_length = np.prod(data_shape)

    # Find the diffrence between the shape of the data and the kernel.
    shape_diff = data_shape - kernel_shape

    if np.any(shape_diff < 0):
        raise ValueError('Kernel shape must be less than or equal to the '
                         'data shape')

    # Set the kernel radius.
    kernel_rad = kernel_shape // 2

    # Rotate, pad and roll the input kernel.
    kernel_rot = np.pad(np.rot90(kernel, 2), ((0, shape_diff[0]),
                        (0, shape_diff[1])), 'constant')
    kernel_rot = np.roll(np.roll(kernel_rot, -kernel_rad[1], axis=1),
                         -kernel_rad[0], axis=0)

    return np.array([np.roll(np.roll(kernel_rot, i, axis=0), j,
                    axis=1).reshape(vec_length) for i in range(data_shape[0])
                    for j in range(data_shape[1])])
