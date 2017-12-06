# -*- coding: utf-8 -*-

"""IMAGE STAMP SELECTION ROUTINES

This module contains methods for selecting stamps or patches from
images.

:Author: Samuel Farrens <samuel.farrens@gmail.com>

:Version: 1.0

:Date: 20/10/2017

"""

from __future__ import division
from builtins import zip
import numpy as np
from itertools import product
from modopt.base.np_adjust import pad2d


def patch_centres(data_shape, layout):
    """Image centres

    This method inds the centres of the patches in a 2D map.

    Parameters
    ----------
    data_shape : tuple
        Shape of the 2D map
    layout : tuple
        Layout of the patches

    Returns
    -------
    np.ndarray array of patch centres

    """

    data_size = np.array(data_size)
    layout = np.array(layout)

    if data_size.size != 2:
        raise ValueError('The the data shape must be of size 2.')
    if layout.size != 2:
        raise ValueError('The the layout must be of size 2.')

    ranges = np.array(list(product(*np.array([np.arange(x) for x in layout]))))
    patch_shape = data_shape // layout
    patch_centre = patch_shape // 2

    return patch_centre + patch_size * ranges


def postage_stamp(data, pos, pixel_rad):
    """Postage stamp

    This metho selects a postage stamp of a given size from a 2D-array.

    Parameters
    ----------
    data : np.ndarray
        Input data array
    pos : tuple
        Position of postage stamp centre
    pixel_rad : tuple
        Pixel radius in each dimension (i.e. size of the stamp from centre)

    Returns
    -------
    np.ndarray rearanged matrix of kernel elements

    Notes
    -----
    The image edges are padded with zeros.

    """

    # Check that the input array has two dimensions.
    data = np.array(data)
    if data.ndim != 2:
        raise ValueError('The input array must be 2D.')

    # If the pixel radius size is one repat the value for the 2nd dimension.
    pixel_rad = np.array(pixel_rad)
    if pixel_rad.size == 1:
        pixel_rad = np.repeat(pixel_rad, 2)
    if pixel_rad.size not in (1, 2):
        raise ValueError('The pixel radius must have a size of 1 or 2.')

    # Check that the array position has a size of two.
    pos = np.array(pos) % np.array(data.shape)
    if pos.size != 2:
        raise ValueError('The array position must have a size of 2.')

    # Check if the pixel radius is within the bounds of the input array.
    if (np.any(pixel_rad < 1) or np.any(np.array(data.shape) // 2 -
                                        pixel_rad < 0)):
        raise ValueError('The pixel radius values must have a value of at '
                         'least 1 and at most half the size of the input '
                         'array. Array size: ' + str(data.shape))

    # Return postage stamp.
    return pad2d(data, pixel_rad)[[slice(a, a + 2 * b + 1) for a, b in
                                   zip(pos, pixel_rad)]]


def pixel_pos(array_shape):
    """Pixel positions

    This method returns all of the pixel positions from a 2D-array.

    Parameters
    ----------
    array_shape : tuple
        Shape of array

    Returns
    -------
    list of pixel positions

    """

    ranges = np.array([np.arange(x) for x in np.array(array_shape)])

    return list(product(*ranges))


class FetchStamps(object):
    """Fetch postage stamps

    This class returns a stack of postage stamps from a given 2D image array.

    Parameters
    ----------
    data : np.ndarray
        Input 2D data array
    pixel_rad : tuple
        Pixel radius in each dimension
    all : boolean, optional
        Option to select all pixels. Default (all=False)

    """

    def __init__(self, data, pixel_rad, all=False):
        self.data = np.array(data)
        self.shape = np.array(self.data.shape)
        self.pixel_rad = np.array(pixel_rad)
        self._check_inputs()
        self._pad_data()
        if all:
            self.n_pixels()

    def _check_inputs(self):
        """Check inputs

        This method checks the class variable values.

        Raises
        ------
        ValueError
            For invalid array dimensions or pixel radius values

        """

        if self.data.ndim != 2:
            raise ValueError('The input array must be 2D.')
        if self.pixel_rad.size == 1:
            self.pixel_rad = np.repeat(self.pixel_rad, 2)
        elif self.pixel_rad.size not in (1, 2):
            raise ValueError('The pixel radius must have a size of 1 or 2.')

    def _pad_data(self):
        """Pad data

        This method pads the input array with zeros.

        """

        self.pad_data = pad2d(self.data, self.pixel_rad)

    def _adjust_pixels(self):
        """Adjust pixels

        This method adjusts the pixel positions according to the pixel radius.

        """

        self.pixels = self.pixels % self.shape + self.pixel_rad

    def get_pixels(self, pixels):
        """Get pixels

        This method gets the desired pixel positions.

        Parameters
        ----------
        pixels : list or np.ndarray
            List of pixel positions

        Raises
        ------
        ValueError
            For invalid number of dimensions for pixel position array

        """

        self.pixels = np.array(pixels)
        if not 1 <= self.pixels.ndim <= 2:
            raise ValueError('Invalid number of dimensions for pixels')
        elif self.pixels.ndim == 2 and self.pixels.shape[1] != 2:
            raise ValueError('The second dimension of pixels must have size 2')
        self._adjust_pixels()

    def n_pixels(self, n_pixels=None, random=False):
        """Number of pixels

        This method selects a specified number of pixel positions.

        Parameters
        ----------
        n_pixels : int, optional
            Number of pixels to keep. Default (n_pixels=None)
        random : bool, optional
            Option to select random pixel position

        """

        self.pixels = pixel_pos(self.shape)
        if random:
            np.random.shuffle(self.pixels)
        self.pixels = self.pixels[:n_pixels]
        self._adjust_pixels()

    def _stamp(self, pos, func=None, *args):
        """Stamp

        This method retrieves a postage stamp from the padded input data at a
        given position.

        Parameters
        ----------
        pos : tuple
            Pixel position in 2D padded array.
        func : function, optional
            Optional function to be applied to postage stamp.

        Returns
        -------
        np.ndarray postage stamp array or result of func

        Raises
        ------
        ValueError
            For for invalid size of pixel position.

        """

        pos = np.array(pos)
        if pos.size != 2:
            raise ValueError('The pixel position must have a size of 2.')
        stamp = self.pad_data[[slice(a - b, a + b + 1) for a, b in
                              zip(pos, self.pixel_rad)]]
        if isinstance(func, type(None)):
            return stamp
        else:
            return func(stamp, *args)

    def scan(self, func=None, *args, **kwargs):
        """Scan stamps

        This method scans the 2D padded input array and retrieves the postage
        stamps at all the desired pixel positions.

        Parameters
        ----------
        func : function, optional
            Optional function to be applied to postage stamps.

        Returns
        -------
        np.ndarray postage stamp arrays or results of func

        """

        if 'arg_type' in kwargs and kwargs['arg_type'] == 'list':
            return np.array([self._stamp(pos, func, arg) for pos, arg in
                             zip(self.pixels, *args)])

        else:
            return np.array([self._stamp(pos, func, *args) for pos in
                             self.pixels])
