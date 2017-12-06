# -*- coding: utf-8 -*-

"""POSITIVITY

This module contains a function that retains only positive coefficients in
an array

:Author: Samuel Farrens <samuel.farrens@gmail.com>

:Version: 1.0

:Date: 26/07/2017

"""

import numpy as np


def positive(data):
    """Positivity operator

    This method preserves only the positive coefficients of the input data, all
    negative coefficients are set to zero

    Parameters
    ----------
    data : np.ndarray, list or tuple
        Input data array

    Returns
    -------
    np.ndarray array with only positive coefficients

    """

    def pos_recursive(data):

        data = np.array(data)

        if not data.dtype == 'O':

            result = list(data * (data > 0))

        else:

            result = [pos_recursive(x) for x in data]

        return result

    return np.array(pos_recursive(data))
