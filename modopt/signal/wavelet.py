# -*- coding: utf-8 -*-

"""WAVELET MODULE

This module contains methods for performing wavelet transformations using iSAP

:Author: Samuel Farrens <samuel.farrens@gmail.com>

:Version: 1.2

:Date: 20/10/2017

"""

from __future__ import division
from builtins import zip
import numpy as np
from os import remove
from subprocess import check_call
from datetime import datetime
from astropy.io import fits
from modopt.image.convolve import convolve
from modopt.base.np_adjust import rotate_stack


def call_mr_transform(data, opt=None, path='./', remove_files=True):
    """Call mr_transform

    This method calls the iSAP module mr_transform

    Parameters
    ----------
    data : np.ndarray
        Input data, 2D array
    opt : list, optional
        List of additonal mr_transform options
    path : str, optional
        Path for output files (default is './')
    remove_files : bool, optional
        Option to remove output files (default is 'True')

    Returns
    -------
    np.ndarray results of transform

    """

    # Create a unique string using the current date and time.
    unique_string = datetime.now().strftime('%Y.%m.%d_%H.%M.%S')

    # Set the ouput file names.
    file_name = path + 'mr_temp_' + unique_string
    file_fits = file_name + '.fits'
    file_mr = file_name + '.mr'

    # Write the input data to a fits file.
    fits.writeto(file_fits, data)

    # Call mr_transform.
    if isinstance(opt, type(None)):
        check_call(['mr_transform', file_fits, file_mr])
    else:
        check_call(['mr_transform'] + opt + [file_fits, file_mr])

    # Retrieve wavelet transformed data.
    result = fits.getdata(file_mr)

    # Return the mr_transform results (and the output file names).
    if remove_files:
        remove(file_fits)
        remove(file_mr)
        return result
    else:
        return result, file_mr


def get_mr_filters(data_shape, opt=None, coarse=False):
    """Get mr_transform filters

    This method obtains wavelet filters by calling mr_transform

    Parameters
    ----------
    data_shape : tuple
        2D data shape
    opt : list, optional
        List of additonal mr_transform options
    coarse : bool, optional
        Option to keep coarse scale (default is 'False')

    Returns
    -------
    np.ndarray 3D array of wavelet filters

    """

    # Adjust the shape of the input data.
    data_shape = np.array(data_shape)
    data_shape += data_shape % 2 - 1

    # Create fake data.
    fake_data = np.zeros(data_shape)
    fake_data[list(zip(data_shape // 2))] = 1

    # Call mr_transform.
    mr_filters = call_mr_transform(fake_data, opt=opt)

    # Return filters
    if coarse:
        return mr_filters
    else:
        return mr_filters[:-1]


def filter_convolve(data, filters, filter_rot=False):
    """Filter convolve

    This method convolves the input image with the wavelet filters

    Parameters
    ----------
    data : np.ndarray
        Input data, 2D array
    filters : np.ndarray
        Wavelet filters, 3D array
    filter_rot : bool, optional
        Option to rotate wavelet filters (default is 'False')

    Returns
    -------
    np.ndarray convolved data

    """

    if filter_rot:
        return np.sum((convolve(coef, f) for coef, f in
                      zip(data, rotate_stack(filters))), axis=0)

    else:
        return np.array([convolve(data, f) for f in filters])


def filter_convolve_stack(data, filters, filter_rot=False):
    """Filter convolve

    This method convolves the a stack of input images with the wavelet filters

    Parameters
    ----------
    data : np.ndarray
        Input data, 3D array
    filters : np.ndarray
        Wavelet filters, 3D array
    filter_rot : bool, optional
        Option to rotate wavelet filters (default is 'False')

    Returns
    -------
    np.ndarray convolved data

    """

    # Return the convolved data cube.
    return np.array([filter_convolve(x, filters, filter_rot=filter_rot)
                     for x in data])
