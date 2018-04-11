# -*- coding: utf-8 -*-

"""REWEIGHTING CLASSES

This module contains classes for reweighting optimisation implementations

:Author: Samuel Farrens <samuel.farrens@cea.fr>

"""

from __future__ import division, print_function
import numpy as np
from modopt.base.types import check_float


class cwbReweight(object):
    r"""Candes, Wakin and Boyd reweighting class

    This class implements the reweighting scheme described in [CWB2007]_

    Parameters
    ----------
    weights : np.ndarray
        Array of weights
    thresh_factor : float
        Threshold factor

    Examples
    --------
    >>> from modopt.signal.reweight import cwbReweight
    >>> a = np.arange(9).reshape(3, 3).astype(float) + 1
    >>> rw = cwbReweight(a)
    >>> rw.weights
    array([[ 1.,  2.,  3.],
           [ 4.,  5.,  6.],
           [ 7.,  8.,  9.]])
    >>> rw.reweight(a)
    >>> rw.weights
    array([[ 0.5,  1. ,  1.5],
           [ 2. ,  2.5,  3. ],
           [ 3.5,  4. ,  4.5]])

    """

    def __init__(self, weights, thresh_factor=1.0):

        self.weights = check_float(weights)
        self.original_weights = np.copy(self.weights)
        self.thresh_factor = check_float(thresh_factor)
        self._rw_num = 1

    def reweight(self, data):
        r"""Reweight

        This method implements the reweighting from section 4 in [CWB2007]_

        Notes
        -----

        Reweighting implemented as:

        .. math::

            w = w \left( \frac{1}{1 + \frac{|x^w|}{n \sigma}} \right)

        """

        print(' - Reweighting: {}'.format(self._rw_num))
        self._rw_num += 1

        data = check_float(data)

        if data.shape != self.weights.shape:
            raise ValueError('Input data must have the same shape as the '
                             'initial weights.')

        self.weights *= (1.0 / (1.0 + np.abs(data) / (self.thresh_factor *
                         self.original_weights)))
