# -*- coding: utf-8 -*-

"""UNIT TESTS FOR IMAGE

This module contains unit tests for the modopt.math module.

:Author: Samuel Farrens <samuel.farrens@gmail.com>

"""

import pytest
import numpy as np
import numpy.testing as npt
from unittest import main, TestCase
from modopt.math import *


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


class MatrixTestCase(TestCase):

    def setUp(self):

        self.data1 = np.arange(9).reshape(3, 3)
        self.data2 = np.arange(3)
        np.random.seed(1)
        self.pmInstance = matrix.PowerMethod(lambda x: x.dot(x.T),
                                             self.data1.shape)

    def tearDown(self):

        self.data1 = None
        self.data2 = None

    def test_gram_schmidt(self):

        npt.assert_allclose(matrix.gram_schmidt(self.data1),
                            np.array([[0., 0.4472136, 0.89442719],
                                     [0.91287093, 0.36514837, -0.18257419],
                                     [-1., 0., 0.]]),
                            err_msg='Incorrect Gram-Schmidt')

    def test_nuclear_norm(self):

        npt.assert_almost_equal(matrix.nuclear_norm(self.data1),
                                15.49193338482967,
                                err_msg='Incorrect nuclear norm')

    def test_project(self):

        npt.assert_array_equal(matrix.project(self.data2, self.data2 + 3),
                               np.array([0., 2.8, 5.6]),
                               err_msg='Incorrect projection')

    def test_rot_matrix(self):

        npt.assert_allclose(matrix.rot_matrix(np.pi / 6),
                            np.array([[0.8660254, -0.5],
                                      [0.5, 0.8660254]]),
                            err_msg='Incorrect rotation matrix')

    def test_rotate(self):

        npt.assert_array_equal(matrix.rotate(self.data1, np.pi / 2),
                               np.array([[2, 5, 8], [1, 4, 7], [0, 3, 6]]),
                               err_msg='Incorrect rotation')

    def test_PowerMethod(self):

        npt.assert_almost_equal(self.pmInstance.spec_rad,
                                0.90429242629600837,
                                err_msg='Incorrect spectral radius')

        npt.assert_almost_equal(self.pmInstance.inv_spec_rad,
                                1.1058369736612865,
                                err_msg='Incorrect inverse spectral radius')


class StatsTestCase(TestCase):

    def setUp(self):

        self.data1 = np.arange(9).reshape(3, 3)
        self.data2 = np.arange(18).reshape(2, 3, 3)

    def tearDown(self):

        self.data1 = None

    def test_gaussian_kernel_max(self):

        npt.assert_allclose(stats.gaussian_kernel(self.data1.shape, 1),
                            np.array([[0.36787944, 0.60653066, 0.36787944],
                                      [0.60653066, 1., 0.60653066],
                                      [0.36787944, 0.60653066, 0.36787944]]),
                            err_msg='Incorrect gaussian kernel: max norm')

    def test_gaussian_kernel_sum(self):

        npt.assert_allclose(stats.gaussian_kernel(self.data1.shape, 1,
                            norm='sum'),
                            np.array([[0.07511361, 0.1238414, 0.07511361],
                                      [0.1238414, 0.20417996, 0.1238414],
                                      [0.07511361, 0.1238414, 0.07511361]]),
                            err_msg='Incorrect gaussian kernel: sum norm')

    def test_mad(self):

        npt.assert_equal(stats.mad(self.data1), 2.0,
                         err_msg='Incorrect median absolute deviation')

    def test_mse(self):

        npt.assert_equal(stats.mse(self.data1, self.data1 + 2), 4.0,
                         err_msg='Incorrect mean squared error')

    def test_psnr_starck(self):

        npt.assert_almost_equal(stats.psnr(self.data1, self.data1 + 2),
                                12.041199826559248,
                                err_msg='Incorrect PSNR: starck')

    def test_psnr_wiki(self):

        npt.assert_almost_equal(stats.psnr(self.data1, self.data1 + 2,
                                method='wiki'),
                                42.110203695399477,
                                err_msg='Incorrect PSNR: wiki')

    def test_psnr_stack(self):

        npt.assert_almost_equal(stats.psnr_stack(self.data2, self.data2 + 2),
                                12.041199826559248,
                                err_msg='Incorrect PSNR stack')

    def test_sigma_mad(self):

        npt.assert_almost_equal(stats.sigma_mad(self.data1),
                                2.9651999999999998,
                                err_msg='Incorrect sigma from MAD')


if __name__ == '__main__':
    main(verbosity=2)
