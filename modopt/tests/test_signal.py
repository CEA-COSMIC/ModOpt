# -*- coding: utf-8 -*-

"""UNIT TESTS FOR SIGNAL

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
        self.data2 = (np.array([[-0.01744594, -0.61438865, -0.25602927,
                                -0.26965119, -0.28327311, -0.29689504,
                                -0.31051696, -0.32413888, -0.3377608],
                               [-0.08435304, -0.50397984, -0.32956518,
                                -0.17902053, -0.02847587, 0.12206878,
                                0.27261343, 0.42315808,  0.57370273],
                               [-0.15126014, -0.39357102, 0.89777139,
                                -0.07186782, -0.04150703, -0.01114624,
                                0.01921455, 0.04957534, 0.07993612],
                               [-0.21816724, -0.28316221, -0.087886,
                                0.92583439, -0.06044521, -0.04672482,
                                -0.03300442, -0.01928402, -0.00556363],
                               [-0.28507434, -0.17275339, -0.0735434,
                                -0.0764634, 0.92061661, -0.08230339,
                                -0.08522339, -0.08814338, -0.09106338],
                               [-0.35198144, -0.06234457, -0.05920079,
                                -0.07876118, -0.09832157, 0.88211804,
                                -0.13744235, -0.15700274, -0.17656313],
                               [-0.41888854, 0.04806424, -0.04485819,
                                -0.08105897, -0.11725975, -0.15346054,
                                0.81033868, -0.2258621, -0.26206289],
                               [-0.48579564, 0.15847306, -0.03051558,
                                -0.08335676, -0.13619793, -0.18903911,
                                -0.24188029, 0.70527854, -0.34756264],
                               [-0.55270274, 0.26888188, -0.01617298,
                                -0.08565455, -0.15513612, -0.22461768,
                                -0.29409925, -0.36358082,  0.56693761]]),
                      np.array([42.23492742, 1.10041151]),
                      np.array([[-0.67608034, -0.73682791],
                                [0.73682791, -0.67608034]]))
        self.data3 = np.array([[-1.05426832e-16, 1.00000000e+00],
                               [2.00000000e+00, 3.00000000e+00],
                               [4.00000000e+00, 5.00000000e+00],
                               [6.00000000e+00, 7.00000000e+00],
                               [8.00000000e+00, 9.00000000e+00],
                               [1.00000000e+01, 1.10000000e+01],
                               [1.20000000e+01, 1.30000000e+01],
                               [1.40000000e+01, 1.50000000e+01],
                               [1.60000000e+01, 1.70000000e+01]])
        self.data4 = np.array([[0.49815487, 0.54291537],
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

        npt.assert_equal(svd.find_n_pc(svd.svd(self.data1)[0]), 3,
                         err_msg='Incorrect number of principal components.')

        npt.assert_raises(ValueError, svd.find_n_pc, np.arange(3))

    def test_calculate_svd(self):

        npt.assert_almost_equal(self.svd[0], np.array(self.data2)[0],
                                err_msg='Incorrect SVD calculation: U')

        npt.assert_almost_equal(self.svd[1], np.array(self.data2)[1],
                                err_msg='Incorrect SVD calculation: S')

        npt.assert_almost_equal(self.svd[2], np.array(self.data2)[2],
                                err_msg='Incorrect SVD calculation: V')

    def test_svd_thresh(self):

        npt.assert_almost_equal(svd.svd_thresh(self.data1), self.data3,
                                err_msg='Incorrect SVD tresholding')

        npt.assert_almost_equal(svd.svd_thresh(self.data1, n_pc=1), self.data4,
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
            self.data3 = np.array([[[174., 165., 174.],
                                    [93., 84., 93.],
                                    [174., 165., 174.]],
                                   [[498., 489., 498.],
                                    [417., 408., 417.],
                                    [498., 489., 498.]],
                                   [[822., 813., 822.],
                                    [741., 732., 741.],
                                    [822., 813., 822.]],
                                   [[1146., 1137., 1146.],
                                    [1065., 1056., 1065.],
                                    [1146., 1137., 1146.]]])
            self.data4 = np.array([[14550., 14586., 14550.],
                                   [14874., 14910., 14874.],
                                   [14550., 14586., 14550.]])
            self.data5 = np.array([[[4., 1., 4.],
                                    [13., 10., 13.],
                                    [22., 19., 22.]],
                                   [[13., 10., 13.],
                                    [49., 46., 49.],
                                    [85., 82., 85.]],
                                   [[22., 19., 22.],
                                    [85., 82., 85.],
                                    [148., 145., 148.]]])

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
