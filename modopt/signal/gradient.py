# -*- coding: utf-8 -*-

"""GRADIENT CLASSES

This module contains classses for defining algorithm gradients.
Based on work by Yinghao Ge and Fred Ngole.

:Author: Samuel Farrens <samuel.farrens@gmail.com>

:Version: 1.3

:Date: 19/07/2017

"""


class GradBasic(object):
    """Basic gradient class

    This class defines the basic methods that will be inherited by specific
    gradient classes

    """

    def MtMX(self, x):
        """M^T M X

        This method calculates the action of the transpose of the matrix M on
        the action of the matrix M on the data X

        Parameters
        ----------
        x : np.ndarray
            Input data array

        Returns
        -------
        np.ndarray result

        Notes
        -----
        Calculates  M^T (MX)

        """

        return self.MtX(self.MX(x))

    def get_grad(self, x):
        """Get the gradient step

        This method calculates the gradient step from the input data

        Parameters
        ----------
        x : np.ndarray
            Input data array

        Returns
        -------
        np.ndarray gradient value

        Notes
        -----

        Calculates M^T (MX - Y)

        """

        self.grad = self.MtX(self.MX(x) - self.y)
