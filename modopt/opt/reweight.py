# -*- coding: utf-8 -*-

"""REWEIGHTING CLASSES.

This module contains classes for reweighting optimisation implementations

:Author: Samuel Farrens <samuel.farrens@cea.fr>

"""

import numpy as np

from modopt.base.types import check_float


class cwbReweight(object):
    """Candes, Wakin and Boyd reweighting class.

    This class implements the reweighting scheme described in
    :cite:`candes2007`

    Parameters
    ----------
    weights : numpy.ndarray
        Array of weights
    thresh_factor : float
        Threshold factor (default is ``1.0``)

    Examples
    --------
    >>> import numpy as np
    >>> from modopt.opt.reweight import cwbReweight
    >>> a = np.arange(9).reshape(3, 3).astype(float) + 1
    >>> rw = cwbReweight(a)
    >>> rw.weights
    array([[1., 2., 3.],
           [4., 5., 6.],
           [7., 8., 9.]])
    >>> rw.reweight(a)
    >>> rw.weights
    array([[0.5, 1. , 1.5],
           [2. , 2.5, 3. ],
           [3.5, 4. , 4.5]])

    """

    def __init__(self, weights, thresh_factor=1.0, verbose=False):

        self.weights = check_float(weights)
        self.original_weights = np.copy(self.weights)
        self.thresh_factor = check_float(thresh_factor)
        self._rw_num = 1
        self.verbose = verbose

    def reweight(self, input_data):
        r"""Reweight.

        This method implements the reweighting from section 4 in
        :cite:`candes2007`

        Parameters
        ----------
        input_data : numpy.ndarray
            Input data

        Raises
        ------
        ValueError
            For invalid input shape

        Notes
        -----
        Reweighting implemented as:

        .. math::

            w = w \left( \frac{1}{1 + \frac{|x^w|}{n \sigma}} \right)

        """
        if self.verbose:
            print(' - Reweighting: {0}'.format(self._rw_num))

        self._rw_num += 1

        input_data = check_float(input_data)

        if input_data.shape != self.weights.shape:
            raise ValueError(
                'Input data must have the same shape as the initial weights.',
            )

        thresh_weights = self.thresh_factor * self.original_weights

        self.weights *= np.array(
            1.0 / (1.0 + np.abs(input_data) / (thresh_weights)),
        )
