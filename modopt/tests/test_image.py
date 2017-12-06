# -*- coding: utf-8 -*-

"""UNIT TESTS FOR IMAGE

This module contains unit tests for the modopt.image module.

:Author: Samuel Farrens <samuel.farrens@gmail.com>

:Version: 1.0

:Date: 17/11/2017

"""

import pytest
import numpy as np
import numpy.testing as npt
from unittest import main, TestCase
from modopt.image import *


class ConvolveTestCase(TestCase):

    def setUp(self):

        self.data1 = np.arange(18).reshape(2, 3, 3)
        self.data2 = self.data1 + 1

    def tearDown(self):

        self.data1 = None
        self.data2 = None

    def test_convolve_astropy(self):

        npt.assert_allclose(convolve.convolve(self.data1[0], self.data2[0],
                            method='astropy'),
                            np.array([[210., 201., 210.], [129., 120., 129.],
                                     [210., 201., 210.]]),
                            err_msg='Incorrect convolution: astropy')

    def test_convolve_scipy(self):

        npt.assert_allclose(convolve.convolve(self.data1[0], self.data2[0],
                            method='scipy'),
                            np.array([[14., 35., 38.], [57., 120., 111.],
                                     [110., 197., 158.]]),
                            err_msg='Incorrect convolution: scipy')

    def test_convolve_stack(self):

        npt.assert_allclose(convolve.convolve_stack(self.data1, self.data2),
                            np.array([[[210., 201., 210.],
                                      [129., 120., 129.],
                                      [210., 201., 210.]],
                                     [[1668., 1659., 1668.],
                                      [1587., 1578., 1587.],
                                      [1668., 1659., 1668.]]]),
                            err_msg='Incorrect convolution: stack')

    def test_convolve_stack_rot(self):

        npt.assert_allclose(convolve.convolve_stack(self.data1, self.data2,
                            rot_kernel=True),
                            np.array([[[150., 159., 150.], [231., 240., 231.],
                                      [150., 159., 150.]],
                                     [[1608., 1617., 1608.],
                                      [1689., 1698., 1689.],
                                      [1608., 1617., 1608.]]]),
                            err_msg='Incorrect convolution: stack rot')


class QualityTestCase(TestCase):

    def setUp(self):

        self.data1 = np.arange(25).reshape(5, 5)
        data2 = np.copy(self.data1)
        data2[2] = 0
        self.data2 = data2

    def tearDown(self):

        self.data1 = None
        self.data2 = None

    def test_nmse(self):

        npt.assert_almost_equal(quality.nmse(self.data1, self.data2),
                                0.1489795918367347, err_msg='Incorrect NMSE')

    def test_e_error(self):

        npt.assert_almost_equal(quality.e_error(self.data1, self.data2),
                                0.042727397878588612, err_msg='Incorrect '
                                'ellipticity error')


class ShapeTestCase(TestCase):
    """Ellipticity Test Case

    This class defines a test suite for the ellipticity measurement methods
    in modopt.import.shape.

    """

    def setUp(self):

        self.data = np.arange(25).reshape((5, 5))

    def tearDown(self):

        self.data = None

    def test_ellipticity_chi(self):

        npt.assert_almost_equal(shape.Ellipticity(self.data,
                                ellip_type='chi').e,
                                np.array([-0.20338994, -0.08474559]),
                                err_msg='Incorrect ellipticity: chi')

    def test_ellipticity_epsilon(self):

        npt.assert_almost_equal(shape.Ellipticity(self.data,
                                ellip_type='epsilon').e,
                                np.array([-0.10296018, -0.04289996]),
                                err_msg='Incorrect ellipticity: epsilon')

    def test_ellipticity_atoms(self):

        npt.assert_almost_equal(shape.ellipticity_atoms(self.data),
                                np.array([-0.20338983, -0.08474576]),
                                err_msg='Incorrect ellipticity: atoms')

    def test_shape_project(self):

        npt.assert_array_equal(shape.shape_project((2, 2)),
                               (np.array([[[0.,  0.], [1.,  1.]], [[0.,  1.],
                                [0.,  1.]], [[1.,  1.], [1.,  1.]],
                                [[0.,  1.], [1.,  2.]], [[0., -1.], [1.,  0.]],
                                [[0.,  0.], [0.,  1.]]])),
                               err_msg='Incorrect shape projection')


if __name__ == '__main__':
    main(verbosity=2)
