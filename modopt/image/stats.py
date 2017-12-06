# -*- coding: utf-8 -*-

"""IMAGE STATISTICS ROUTINES

This module contains methods for the statistical analysis of images.

:Author: Samuel Farrens <samuel.farrens@gmail.com>

:Version: 1.1

:Date: 20/10/2017

"""

from __future__ import division
import numpy as np
from modopt.base.np_adjust import pad2d


class SAT():
    """Summed Area Table

    This class produces a Summed Area Table (SAT) for fast and efficient
    statistics on image patches.

    Parameters
    ----------
    data : np.ndarray
        Input 2D data array

    Notes
    -----
    Also know as Itegral Image (i in the class features).

    TODO
    ----
    Add equations and citations

    """

    def __init__(self, data):

        self.x = data
        self.x2 = self.x ** 2
        self.i = self.x.cumsum(axis=0).cumsum(axis=1)
        self.i2 = self.x2.cumsum(axis=0).cumsum(axis=1)
        self.i_pad = pad2d(self.i, 1)
        self.i2_pad = pad2d(self.i2, 1)

    def get_area(self, data, corners):
        """Get area

        This method calculates the area of a patch.

        Parameters
        ----------
        data : np.ndarray
            Input 2D data array
        corners : tuple
            Positions of upper left and bottom right corners.

        Returns
        -------
        float area

        """

        corners = np.array(corners)
        corners[1] += 1

        a = data[zip(corners[0])]
        b = data[corners[0, 0], corners[1, 1]]
        c = data[corners[1, 0], corners[0, 1]]
        d = data[zip(corners[1])]

        return float(a + d - b - c)

    def get_npx(self, corners):
        """Get number of pixels

        This method calculates the number of pixels in a patch.

        Parameters
        ----------
        corners : tuple
            Positions of upper left and bottom right corners.

        Returns
        -------
        int number of pixels

        """

        return np.prod(np.diff(corners, axis=0) + 1)

    def get_var(self):
        """Get variance

        This method calculates the variance and standard deviation of a patch.

        """

        self.var = (self.area2 - (self.area ** 2 / self.npx)) / self.npx
        self.std = np.sqrt(self.var)

    def set_patch(self, corners):
        """Set patch

        This method sets the corner positions of a single patch.

        Parameters
        ----------
        corners : tuple
            Positions of upper left and bottom right corners.

        """

        self.area = self.get_area(self.i_pad, corners)
        self.area2 = self.get_area(self.i2_pad, corners)
        self.npx = self.get_npx(corners)
        self.get_var()

    def set_patches(self, corners):
        """Set patches

        This method sets the corner positions for multiple patches.

        Parameters
        ----------
        corners : list
            List of the positions of upper left and bottom right corners.

        """

        self.area = np.array([self.get_area(self.i_pad, corner)
                              for corner in corners])
        self.area2 = np.array([self.get_area(self.i2_pad, corner)
                               for corner in corners])
        self.npx = np.array([self.get_npx(corner) for corner in corners])
        self.get_var()
