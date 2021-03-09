# -*- coding: utf-8 -*-

"""UNIT TESTS FOR SIGNAL.

This module contains unit tests for the modopt.signal module.

:Author: Samuel Farrens <samuel.farrens@cea.fr>

"""

from unittest import TestCase
import numpy as np
import numpy.testing as npt
from modopt.signal import *


class FilterTestCase(TestCase):

    def test_guassian_filter(self):

        npt.assert_almost_equal(filter.Gaussian_filter(1, 1),
                                0.24197072451914337,
                                err_msg='Incorrect Gaussian filter')

        npt.assert_almost_equal(filter.Gaussian_filter(1, 1, norm=False),
                                0.60653065971263342,
                                err_msg='Incorrect Gaussian filter')

    def test_mex_hat(self):

        npt.assert_almost_equal(filter.mex_hat(2, 1),
                                -0.35213905225713371,
                                err_msg='Incorrect Mexican hat filter')

    def test_mex_hat_dir(self):

        npt.assert_almost_equal(filter.mex_hat_dir(1, 2, 1),
                                0.17606952612856686,
                                err_msg='Incorrect directional Mexican hat '
                                        'filter')


class NoiseTestCase(TestCase):

    def setUp(self):

        self.data1 = np.arange(9).reshape(3, 3).astype(float)
        self.data2 = np.array([[0., 2., 2.], [4., 5., 10.],
                               [11., 15., 18.]])
        self.data3 = np.array([[1.62434536, 0.38824359, 1.47182825],
                               [1.92703138, 4.86540763, 2.6984613],
                               [7.74481176, 6.2387931, 8.3190391]])
        self.data4 = np.array([[0., 0., 0.], [0., 0., 5.], [6., 7., 8.]])
        self.data5 = np.array([[0., 0., 0.], [0., 0., 0.],
                               [1., 2., 3.]])

    def tearDown(self):

        self.data1 = None
        self.data2 = None
        self.data3 = None
        self.data4 = None
        self.data5 = None

    def test_add_noise_poisson(self):

        np.random.seed(1)
        npt.assert_array_equal(noise.add_noise(self.data1,
                               noise_type='poisson'), self.data2,
                               err_msg='Incorrect noise: Poisson')

        npt.assert_raises(ValueError, noise.add_noise, self.data1,
                          noise_type='bla')

        npt.assert_raises(ValueError, noise.add_noise, self.data1, (1, 1))

    def test_add_noise_gaussian(self):

        np.random.seed(1)
        npt.assert_almost_equal(noise.add_noise(self.data1), self.data3,
                                err_msg='Incorrect noise: Gaussian')

        np.random.seed(1)
        npt.assert_almost_equal(noise.add_noise(self.data1,
                                sigma=(1, 1, 1)), self.data3,
                                err_msg='Incorrect noise: Gaussian')

    def test_thresh_hard(self):

        npt.assert_array_equal(noise.thresh(self.data1, 5), self.data4,
                               err_msg='Incorrect threshold: hard')

        npt.assert_raises(ValueError, noise.thresh, self.data1, 5,
                          threshold_type='bla')

    def test_thresh_soft(self):

        npt.assert_array_equal(noise.thresh(self.data1, 5,
                               threshold_type='soft'), self.data5,
                               err_msg='Incorrect threshold: soft')


class PositivityTestCase(TestCase):

    def setUp(self):

        self.data1 = np.arange(9).reshape(3, 3) - 5
        self.data2 = np.array([[0, 0, 0], [0, 0, 0], [1, 2, 3]])
        self.data3 = [np.arange(5) - 3, np.arange(4) - 2]
        self.data4 = np.array([list([0, 0, 0, 0, 1]), list([0, 0, 0, 1])],
                              dtype=object)
        self.err = 'Incorrect positivity'

    def tearDown(self):

        self.data1 = None
        self.data2 = None

    def test_positivity(self):

        npt.assert_equal(positivity.positive(-1), 0, err_msg=self.err)

        npt.assert_equal(positivity.positive(-1.0), -0.0, err_msg=self.err)

        npt.assert_equal(positivity.positive(self.data1), self.data2,
                         err_msg=self.err)

        npt.assert_equal(positivity.positive(self.data3), self.data4,
                         err_msg=self.err)

        npt.assert_raises(TypeError, positivity.positive, '-1')


class SVDTestCase(TestCase):

    def setUp(self):

        self.data1 = np.arange(18).reshape(9, 2).astype(float)
        self.data2 = np.arange(32).reshape(16, 2).astype(float)
        self.data3 = (np.array([[-0.01744594, -0.61438865],
                                [-0.08435304, -0.50397984],
                                [-0.15126014, -0.39357102],
                                [-0.21816724, -0.28316221],
                                [-0.28507434, -0.17275339],
                                [-0.35198144, -0.06234457],
                                [-0.41888854, 0.04806424],
                                [-0.48579564, 0.15847306],
                                [-0.55270274, 0.26888188]]),
                      np.array([42.23492742, 1.10041151]),
                      np.array([[-0.67608034, -0.73682791],
                                [0.73682791, -0.67608034]]))
        self.data4 = np.array([[-1.05426832e-16, 1.00000000e+00],
                               [2.00000000e+00, 3.00000000e+00],
                               [4.00000000e+00, 5.00000000e+00],
                               [6.00000000e+00, 7.00000000e+00],
                               [8.00000000e+00, 9.00000000e+00],
                               [1.00000000e+01, 1.10000000e+01],
                               [1.20000000e+01, 1.30000000e+01],
                               [1.40000000e+01, 1.50000000e+01],
                               [1.60000000e+01, 1.70000000e+01]])
        self.data5 = np.array([[0.49815487, 0.54291537],
                               [2.40863386, 2.62505584],
                               [4.31911286, 4.70719631],
                               [6.22959185, 6.78933678],
                               [8.14007085, 8.87147725],
                               [10.05054985, 10.95361772],
                               [11.96102884, 13.03575819],
                               [13.87150784, 15.11789866],
                               [15.78198684, 17.20003913]])
        self.svd = svd.calculate_svd(self.data1)

    def tearDown(self):

        self.data1 = None
        self.data2 = None
        self.data3 = None
        self.data4 = None
        self.svd = None

    def test_find_n_pc(self):

        npt.assert_equal(svd.find_n_pc(svd.svd(self.data2)[0]), 2,
                         err_msg='Incorrect number of principal components.')

        npt.assert_raises(ValueError, svd.find_n_pc, np.arange(3))

    def test_calculate_svd(self):

        npt.assert_almost_equal(self.svd[0], np.array(self.data3)[0],
                                err_msg='Incorrect SVD calculation: U')

        npt.assert_almost_equal(self.svd[1], np.array(self.data3)[1],
                                err_msg='Incorrect SVD calculation: S')

        npt.assert_almost_equal(self.svd[2], np.array(self.data3)[2],
                                err_msg='Incorrect SVD calculation: V')

    def test_svd_thresh(self):

        npt.assert_almost_equal(svd.svd_thresh(self.data1), self.data4,
                                err_msg='Incorrect SVD tresholding')

        npt.assert_almost_equal(svd.svd_thresh(self.data1, n_pc=1), self.data5,
                                err_msg='Incorrect SVD tresholding')

        npt.assert_almost_equal(svd.svd_thresh(self.data1, n_pc='all'),
                                self.data1,
                                err_msg='Incorrect SVD tresholding')

        npt.assert_raises(TypeError, svd.svd_thresh, 1)

        npt.assert_raises(ValueError, svd.svd_thresh, self.data1, n_pc='bla')

    def test_svd_thresh_coef(self):

        npt.assert_almost_equal(svd.svd_thresh_coef(self.data1,
                                lambda x: x, 0),
                                self.data1,
                                err_msg='Incorrect SVD coefficient '
                                        'tresholding')

        npt.assert_raises(TypeError, svd.svd_thresh_coef, self.data1, 0, 0)


class ValidationTestCase(TestCase):

    def setUp(self):

        self.data1 = np.arange(9).reshape(3, 3).astype(float)

    def tearDown(self):

        self.data1 = None

    def test_transpose_test(self):

        np.random.seed(2)
        npt.assert_equal(validation.transpose_test(lambda x, y: x.dot(y),
                         lambda x, y: x.dot(y.T), self.data1.shape,
                         x_args=self.data1), None)

        npt.assert_raises(TypeError, validation.transpose_test, 0, 0,
                          self.data1.shape, x_args=self.data1)


class WaveletTestCase(TestCase):

    def setUp(self):

        self.data1 = np.arange(9).reshape(3, 3).astype(float)
        self.data2 = np.arange(36).reshape(4, 3, 3).astype(float)
        self.data3 = np.array([[[6., 20., 26.],
                                [36., 84., 84.],
                                [90., 164., 134.]],
                               [[78., 155., 134.],
                                [225., 408., 327.],
                                [270., 461., 350.]],
                               [[150., 290., 242.],
                                [414., 732., 570.],
                                [450., 758., 566.]],
                               [[222., 425., 350.],
                                [603., 1056., 813.],
                                [630., 1055., 782.]]])

        self.data4 = np.array([[6496., 9796., 6544.],
                               [9924., 14910., 9924.],
                               [6544., 9796., 6496.]])

        self.data5 = np.array([[[0., 1., 4.],
                                [3., 10., 13.],
                                [6., 19., 22.]],
                               [[3., 10., 13.],
                                [24., 46., 40.],
                                [45., 82., 67.]],
                               [[6., 19., 22.],
                                [45., 82., 67.],
                                [84., 145., 112.]]])

    def tearDown(self):

        self.data1 = None
        self.data2 = None
        self.data3 = None
        self.data4 = None
        self.data5 = None

    def test_filter_convolve(self):

        npt.assert_almost_equal(wavelet.filter_convolve(self.data1,
                                self.data2), self.data3,
                                err_msg='Inccorect filter comvolution.')

        npt.assert_almost_equal(wavelet.filter_convolve(self.data2,
                                self.data2, filter_rot=True), self.data4,
                                err_msg='Inccorect filter comvolution.')

    def test_filter_convolve_stack(self):

        npt.assert_almost_equal(wavelet.filter_convolve_stack(self.data1,
                                self.data1), self.data5,
                                err_msg='Inccorect filter stack comvolution.')
