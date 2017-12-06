# -*- coding: utf-8 -*-

"""REWEIGHTING CLASSES

This module contains classes for reweighting optimisation implementations

:Author: Samuel Farrens <samuel.farrens@gmail.com>

:Version: 1.3

:Date: 20/10/2017

"""

from __future__ import division
import numpy as np


class cwbReweight(object):
    r"""Candes, Wakin and Boyd reweighting class

    This class implements the reweighting scheme described in [CWB2007]_

    Parameters
    ----------
    weights : np.ndarray
        Array of weights
    thresh_factor : float
        Threshold factor

    """

    def __init__(self, weights, thresh_factor=1):

        self.weights = weights
        self.original_weights = np.copy(self.weights)
        self.thresh_factor = thresh_factor

    def reweight(self, data):
        r"""Reweight

        This method implements the reweighting from section 4 in [CWB2007]_

        Notes
        -----

        Reweighting implemented as:

        .. math::

            w = w \left( \frac{1}{1 + \frac{|x^w|}{n \sigma}} \right)

        """

        self.weights *= (1.0 / (1.0 + np.abs(data) / (self.thresh_factor *
                         self.original_weights)))
