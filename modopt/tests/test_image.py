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
