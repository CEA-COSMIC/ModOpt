# -*- coding: utf-8 -*-

"""SHAPE ESTIMATION ROUTINES

This module contains methods and classes for estimating galaxy shapes.

:Author: Samuel Farrens <samuel.farrens@gmail.com>

:Version: 1.4

:Date: 20/10/2017

Notes
-----
Some of the methods in this module are based on work by Fred Ngole.

"""

from __future__ import division
import numpy as np


def ellipticity_atoms(data, offset=0):
    r"""Calculate ellipticity

    This method calculates the ellipticity of an image from its shape
    projection components.

    Parameters
    ----------
    data : np.ndarray
        Input data array, the image to be analysed
    offset : int, optional
        Shape projection offset (default is '0')

    Returns
    -------
    np.ndarray of the image ellipticity components

    See Also
    --------
    shape_project : shape projection matrix

    Notes
    -----
    This technique was developed by Fred Ngole and implements the following
    equations:

        - Equations C.1 and C.2 from [NS2016]_ appendix:

        .. math::

            e_1(\mathbf{X}_i) = \frac{<\mathbf{X}_i, \mathbf{U}_4>
                                      <\mathbf{X}_i, \mathbf{U}_2> -
                                      <\mathbf{X}_i, \mathbf{U}_0>^2 +
                                      <\mathbf{X}_i, \mathbf{U}_1>^2}
                                      {<\mathbf{X}_i, \mathbf{U}_3>
                                      <\mathbf{X}_i, \mathbf{U}_2> -
                                      <\mathbf{X}_i, \mathbf{U}_0>^2 -
                                      <\mathbf{X}_i, \mathbf{U}_1>^2
                                      }

            e_2(\mathbf{X}_i) = \frac{2\left(<\mathbf{X}_i, \mathbf{U}_5>
                                      <\mathbf{X}_i, \mathbf{U}_2> -
                                      <\mathbf{X}_i, \mathbf{U}_0>
                                      <\mathbf{X}_i, \mathbf{U}_1>\right)}
                                      {<\mathbf{X}_i, \mathbf{U}_3>
                                      <\mathbf{X}_i, \mathbf{U}_2> -
                                      <\mathbf{X}_i, \mathbf{U}_0>^2 -
                                      <\mathbf{X}_i, \mathbf{U}_1>^2
                                      }

    Examples
    --------
    >>> from image.shape import ellipticity_atoms
    >>> import numpy as np
    >>> a = np.zeros((5, 5))
    >>> a[2, 1:4] += 1
    >>> ellipticity_atoms(a)
    array([-1.,  0.])

    >>> b = np.zeros((5, 5))
    >>> b[1:4, 2] += 1
    >>> ellipticity_atoms(b)
    array([ 1.,  0.])

    """

    XU = [np.sum(data * U) for U in shape_project(data.shape, offset)]

    divisor = XU[3] * XU[2] - XU[0] ** 2 - XU[1] ** 2
    e1 = (XU[4] * XU[2] - XU[0] ** 2 + XU[1] ** 2) / divisor
    e2 = 2 * (XU[5] * XU[2] - XU[0] * XU[1]) / divisor

    return np.array([e1, e2])


def shape_project(shape, offset=0, return_norm=False):
    r"""Shape projection matrix

    This method generates a shape projection matrix for a given image.

    Parameters
    ----------
    shape : list, tuple or np.ndarray
        List of image dimensions
    offset : int, optional
        Shape projection offset (default is '0')
    return_norm : bool, optional
        Option to return l2 normalised shape projection components
        (default is 'False')

    Returns
    -------
    np.ndarray of shape projection components

    See Also
    --------
    ellipticity_atoms : calculate ellipticity

    Notes
    -----
    This technique was developed by Fred Ngole and implements the following
    equations:

        - Equations from [NS2016]_ appendix:

        .. math::

            U_1 &= (k)_{1 \leq k \leq N_l, 1 \leq l \leq N_c} \\
            U_2 &= (l)_{1 \leq k \leq N_l, 1 \leq l \leq N_c} \\
            U_3 &= (1)_{1 \leq k \leq N_l, 1 \leq l \leq N_c} \\
            U_4 &= (k^2 + l^2)_{1 \leq k \leq N_l, 1 \leq l \leq N_c} \\
            U_5 &= (k^2 - l^2)_{1 \leq k \leq N_l, 1 \leq l \leq N_c} \\
            U_6 &= (kl)_{1 \leq k \leq N_l, 1 \leq l \leq N_c}

    Examples
    --------
    >>> from image.shape import shape_project
    >>> shape_project([3, 3])
    array([[[ 0.,  0.,  0.],
            [ 1.,  1.,  1.],
            [ 2.,  2.,  2.]],
    <BLANKLINE>
           [[ 0.,  1.,  2.],
            [ 0.,  1.,  2.],
            [ 0.,  1.,  2.]],
    <BLANKLINE>
           [[ 1.,  1.,  1.],
            [ 1.,  1.,  1.],
            [ 1.,  1.,  1.]],
    <BLANKLINE>
           [[ 0.,  1.,  4.],
            [ 1.,  2.,  5.],
            [ 4.,  5.,  8.]],
    <BLANKLINE>
           [[ 0., -1., -4.],
            [ 1.,  0., -3.],
            [ 4.,  3.,  0.]],
    <BLANKLINE>
           [[ 0.,  0.,  0.],
            [ 0.,  1.,  2.],
            [ 0.,  2.,  4.]]])

    """

    U = []
    U.append(np.outer(np.arange(shape[0]) + offset, np.ones(shape[1])))
    U.append(np.outer(np.ones(shape[0]), np.arange(shape[1]) + offset))
    U.append(np.ones(shape))
    U.append(U[0] ** 2 + U[1] ** 2)
    U.append(U[0] ** 2 - U[1] ** 2)
    U.append(U[0] * U[1])

    if return_norm:
        np.array([np.linalg.norm(x, 2) for x in U])
    else:
        return np.array(U)


class Ellipticity():
    """ Image ellipticity class

    This class calculates image ellipticities from quadrupole moments.

    Parameters
    ----------
    data : np.ndarray
        Input data array, the image to be analysed
    sigma : int, optional
        Estimation error (default is '1000')
    centroid : np.ndarray, optional
        Centroid positions [x, y] of the input image (defualt is 'None')
    moments : np.ndarray, optional
        Quadrupole moments [[q00, q01], [q10, q11]] of the input image
        (defualt is 'None')
    ellip_type : str {'chi', 'epsilon'}, optional
        Ellipticity type (default is 'chi')

    Examples
    --------
    >>> from image.shape import Ellipticity
    >>> import numpy as np
    >>> a = np.zeros((5, 5))
    >>> a[2, 1:4] += 1
    >>> Ellipticity(a).e
    array([-1.,  0.])

    >>> b = np.zeros((5, 5))
    >>> b[1:4, 2] += 1
    >>> Ellipticity(b).e
    array([ 1.,  0.])

    """

    def __init__(self, data, sigma=1000, centroid=None, moments=None,
                 ellip_type='chi'):

        self._data = data
        self._sigma = sigma
        self._ranges = np.array([np.arange(i) for i in data.shape])
        self._ellip_type = ellip_type
        self._check_ellip_type()

        if not isinstance(moments, type(None)):
            self.moments = np.array(moments).astype('complex').reshape(2, 2)
            self._get_ellipse()
        elif isinstance(centroid, type(None)):
            self._get_centroid()
        else:
            self.centroid = centroid
            self._update_weights()
            self._get_moments()

    def _check_ellip_type(self):
        """Check Ellipticity Type

        This method raises an error if ellip_type is not 'chi' or 'epsilon'.

        Raises
        ------
        ValueError for invalid ellip_type

        """

        if self._ellip_type not in ('chi', 'epsilon'):
            raise ValueError('Invalid ellip_type, options are "chi" or '
                             '"epsilon"')

    def _update_xy(self):
        """Update the x and y values

        This method updates the values of x and y using the current centroid.

        """

        self._x = np.outer(self._ranges[0] - self.centroid[0],
                           np.ones(self._data.shape[1]))
        self._y = np.outer(np.ones(self._data.shape[0]),
                           self._ranges[1] - self.centroid[1])

    def _update_weights(self):
        """Update the weights

        This method updates the value of the weights using the current values
        of x and y.

        Notes
        -----
        This method implements the following equations:

            - The exponential part of equation 1 from [BM2007]_ to calculate
              the weights:

            .. math::

                w(x,y) = e^{-\\frac{\\left((x-x_c)^2+(y-y_c)^2\\right)}
                    {2\\sigma^2}}

        """

        self._update_xy()
        self._weights = np.exp(-(self._x ** 2 + self._y ** 2) /
                               (2 * self._sigma ** 2))

    def _update_centroid(self):
        """Update the centroid

        This method updates the centroid value using the current weights.

        Notes
        -----
        This method implements the following equations:

            - Equation 2a, 2b and 2c from [BM2007]_ to calculate the position
              moments:

            .. math::

                S_w = \sum_{x,y} I(x,y)w(x,y)

                S_x = \sum_{x,y} xI(x,y)w(x,y)

                S_y = \sum_{x,y} yI(x,y)w(x,y)

            - Equation 3 from [BM2007]_ to calculate the centroid:

            .. math::

                X_c = S_x/S_w,\\
                Y_c = S_y/S_w

        """

        # Calculate the position moments.
        iw = np.array([np.sum(self._data * self._weights, axis=i)
                       for i in (1, 0)])
        sw = np.sum(iw, axis=1)
        sxy = np.sum(iw * self._ranges, axis=1)

        # Update the centroid value.
        self.centroid = sxy / sw

    def _get_centroid(self, n_iter=10):
        """Calculate centroid

        This method iteratively calculates the centroid of the image.

        Parameters
        ----------
        n_iter : int, optional
            Number of iterations (deafult is '10')

        """

        # Set initial value for the weights.
        self._weights = np.ones(self._data.shape)

        # Iteratively calculate the centroid.
        for i in range(n_iter):

            # Update the centroid value.
            self._update_centroid()

            # Update the weights.
            self._update_weights()

        # Calculate the quadrupole moments.
        self._get_moments()

    def _get_moments(self):
        """ Calculate the quadrupole moments

        This method calculates the quadrupole moments.

        Notes
        -----
        This method implements the following equations:

            - Equation 10 from [C2013]_ to calculate the moments:

            .. math::

                Q_{ij}=\\frac{\\int\\int\\Phi(x_i,x_j) w(x_i,x_j)
                    (x_i-\\bar{x_i})(x_j-\\bar{x_j}) dx_idx_j}
                    {\\int\\int\\Phi(x_i,x_j)w(x_i,x_j)dx_idx_j}

        """

        # Calculate moments.
        q = np.array([np.sum(self._data * self._weights * xi * xj) for xi in
                      (self._x, self._y) for xj in (self._x, self._y)])

        self.moments = (q / np.sum(self._data *
                        self._weights)).reshape(2, 2).astype('complex')

        # Calculate the ellipticities.
        self._get_ellipse()

    def _get_ellipse(self):
        """Calculate the ellipticities

        This method cacluates ellipticities from quadrupole moments.

        Notes
        -----
        This method implements the following equations:

            - Equation 11 from [C2013]_ to calculate the size:

            .. math:: R^2 = Q_{00} + Q_{11}

            - Equation 7 from [S2005]_ to calculate the ellipticities:

            .. math::

               \\chi = \\frac{Q_{00}-Q_{11}+iQ_{01}+iQ_{10}}{R^2}

               \\epsilon = \\frac{Q_{00}-Q_{11}+iQ_{01}+iQ_{10}}{R^2 +
                   2\\sqrt{(Q_{00}Q_{11} - Q_{01}Q_{10})}}

        """

        # Calculate the size.
        self.r2 = self.moments[0, 0] + self.moments[1, 1]

        # Calculate the numerator
        numerator = (self.moments[0, 0] - self.moments[1, 1] + np.complex(0,
                     self.moments[0, 1] + self.moments[1, 0]))

        # Calculate the denominator
        if self._ellip_type == 'epsilon':
            denominator = (self.r2 + 2 * np.sqrt(self.moments[0, 0] *
                           self.moments[1, 1] - self.moments[0, 1] *
                           self.moments[1, 0]))

        else:
            denominator = self.r2

        # Calculate the ellipticity
        ellip = numerator / denominator

        self.e = np.array([ellip.real, ellip.imag])
