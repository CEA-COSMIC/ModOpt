# -*- coding: utf-8 -*-

"""UNIT TESTS FOR SIGNAL.

This module contains unit tests for the modopt.signal module.

:Author: Samuel Farrens <samuel.farrens@cea.fr>

"""

from unittest import TestCase

import numpy as np
import numpy.testing as npt

from modopt.signal import filter, noise, positivity, svd, validation, wavelet


class FilterTestCase(TestCase):
    """Test case for filter module."""

    def test_guassian_filter(self):
        """Test guassian_filter."""
        npt.assert_almost_equal(
            filter.gaussian_filter(1, 1),
            0.24197072451914337,
            err_msg='Incorrect Gaussian filter',
        )

        npt.assert_almost_equal(
            filter.gaussian_filter(1, 1, norm=False),
            0.60653065971263342,
            err_msg='Incorrect Gaussian filter',
        )

    def test_mex_hat(self):
        """Test mex_hat."""
        npt.assert_almost_equal(
            filter.mex_hat(2, 1),
            -0.35213905225713371,
            err_msg='Incorrect Mexican hat filter',
        )

    def test_mex_hat_dir(self):
        """Test mex_hat_dir."""
        npt.assert_almost_equal(
            filter.mex_hat_dir(1, 2, 1),
            0.17606952612856686,
            err_msg='Incorrect directional Mexican hat filter',
        )


class NoiseTestCase(TestCase):
    """Test case for noise module."""

    def setUp(self):
        """Set test parameter values."""
        self.data1 = np.arange(9).reshape(3, 3).astype(float)
        self.data2 = np.array(
            [[0, 2.0, 2.0], [4.0, 5.0, 10], [11.0, 15.0, 18.0]],
        )
        self.data3 = np.array([
            [1.62434536, 0.38824359, 1.47182825],
            [1.92703138, 4.86540763, 2.6984613],
            [7.74481176, 6.2387931, 8.3190391],
        ])
        self.data4 = np.array([[0, 0, 0], [0, 0, 5.0], [6.0, 7.0, 8.0]])
        self.data5 = np.array(
            [[0, 0, 0], [0, 0, 0], [1.0, 2.0, 3.0]],
        )

    def tearDown(self):
        """Unset test parameter values."""
        self.data1 = None
        self.data2 = None
        self.data3 = None
        self.data4 = None
        self.data5 = None

    def test_add_noise_poisson(self):
        """Test add_noise with Poisson noise."""
        np.random.seed(1)
        npt.assert_array_equal(
            noise.add_noise(self.data1, noise_type='poisson'),
            self.data2,
            err_msg='Incorrect noise: Poisson',
        )

        npt.assert_raises(
            ValueError,
            noise.add_noise,
            self.data1,
            noise_type='bla',
        )

        npt.assert_raises(ValueError, noise.add_noise, self.data1, (1, 1))

    def test_add_noise_gaussian(self):
        """Test add_noise with Gaussian noise."""
        np.random.seed(1)
        npt.assert_almost_equal(
            noise.add_noise(self.data1),
            self.data3,
            err_msg='Incorrect noise: Gaussian',
        )

        np.random.seed(1)
        npt.assert_almost_equal(
            noise.add_noise(self.data1, sigma=(1, 1, 1)),
            self.data3,
            err_msg='Incorrect noise: Gaussian',
        )

    def test_thresh_hard(self):
        """Test thresh with hard threshold."""
        npt.assert_array_equal(
            noise.thresh(self.data1, 5),
            self.data4,
            err_msg='Incorrect threshold: hard',
        )

        npt.assert_raises(
            ValueError,
            noise.thresh,
            self.data1,
            5,
            threshold_type='bla',
        )

    def test_thresh_soft(self):
        """Test thresh with soft threshold."""
        npt.assert_array_equal(
            noise.thresh(self.data1, 5, threshold_type='soft'),
            self.data5,
            err_msg='Incorrect threshold: soft',
        )


class PositivityTestCase(TestCase):
    """Test case for positivity module."""

    def setUp(self):
        """Set test parameter values."""
        self.data1 = np.arange(9).reshape(3, 3) - 5
        self.data2 = np.array([[0, 0, 0], [0, 0, 0], [1, 2, 3]])
        self.data3 = np.array(
            [np.arange(5) - 3, np.arange(4) - 2],
            dtype=object,
        )
        self.data4 = np.array(
            [np.array([0, 0, 0, 0, 1]), np.array([0, 0, 0, 1])],
            dtype=object,
        )
        self.pos_dtype_obj = positivity.positive(self.data3)
        self.err = 'Incorrect positivity'

    def tearDown(self):
        """Unset test parameter values."""
        self.data1 = None
        self.data2 = None

    def test_positivity(self):
        """Test positivity."""
        npt.assert_equal(positivity.positive(-1), 0, err_msg=self.err)

        npt.assert_equal(
            positivity.positive(-1.0),
            -float(0),
            err_msg=self.err,
        )

        npt.assert_equal(
            positivity.positive(self.data1),
            self.data2,
            err_msg=self.err,
        )

        for expected, output in zip(self.data4, self.pos_dtype_obj):
            print(expected, output)
            npt.assert_array_equal(expected, output, err_msg=self.err)

        npt.assert_raises(TypeError, positivity.positive, '-1')


class SVDTestCase(TestCase):
    """Test case for svd module."""

    def setUp(self):
        """Set test parameter values."""
        self.data1 = np.arange(18).reshape(9, 2).astype(float)
        self.data2 = np.arange(32).reshape(16, 2).astype(float)
        self.data3 = np.array(
            [
                np.array([
                    [-0.01744594, -0.61438865],
                    [-0.08435304, -0.50397984],
                    [-0.15126014, -0.39357102],
                    [-0.21816724, -0.28316221],
                    [-0.28507434, -0.17275339],
                    [-0.35198144, -0.06234457],
                    [-0.41888854, 0.04806424],
                    [-0.48579564, 0.15847306],
                    [-0.55270274, 0.26888188],
                ]),
                np.array([42.23492742, 1.10041151]),
                np.array([
                    [-0.67608034, -0.73682791],
                    [0.73682791, -0.67608034],
                ]),
            ],
            dtype=object,
        )
        self.data4 = np.array([
            [-1.05426832e-16, 1.0],
            [2.0, 3.0],
            [4.0, 5.0],
            [6.0, 7.0],
            [8.0, 9.0],
            [1.0e1, 1.1e1],
            [1.2e1, 1.3e1],
            [1.4e1, 1.5e1],
            [1.6e1, 1.7e1],
        ])
        self.data5 = np.array([
            [0.49815487, 0.54291537],
            [2.40863386, 2.62505584],
            [4.31911286, 4.70719631],
            [6.22959185, 6.78933678],
            [8.14007085, 8.87147725],
            [10.05054985, 10.95361772],
            [11.96102884, 13.03575819],
            [13.87150784, 15.11789866],
            [15.78198684, 17.20003913],
        ])
        self.svd = svd.calculate_svd(self.data1)

    def tearDown(self):
        """Unset test parameter values."""
        self.data1 = None
        self.data2 = None
        self.data3 = None
        self.data4 = None
        self.svd = None

    def test_find_n_pc(self):
        """Test find_n_pc."""
        npt.assert_equal(
            svd.find_n_pc(svd.svd(self.data2)[0]),
            2,
            err_msg='Incorrect number of principal components.',
        )

        npt.assert_raises(ValueError, svd.find_n_pc, np.arange(3))

    def test_calculate_svd(self):
        """Test calculate_svd."""
        npt.assert_almost_equal(
            self.svd[0],
            np.array(self.data3)[0],
            err_msg='Incorrect SVD calculation: U',
        )

        npt.assert_almost_equal(
            self.svd[1],
            np.array(self.data3)[1],
            err_msg='Incorrect SVD calculation: S',
        )

        npt.assert_almost_equal(
            self.svd[2],
            np.array(self.data3)[2],
            err_msg='Incorrect SVD calculation: V',
        )

    def test_svd_thresh(self):
        """Test svd_thresh."""
        npt.assert_almost_equal(
            svd.svd_thresh(self.data1),
            self.data4,
            err_msg='Incorrect SVD tresholding',
        )

        npt.assert_almost_equal(
            svd.svd_thresh(self.data1, n_pc=1),
            self.data5,
            err_msg='Incorrect SVD tresholding',
        )

        npt.assert_almost_equal(
            svd.svd_thresh(self.data1, n_pc='all'),
            self.data1,
            err_msg='Incorrect SVD tresholding',
        )

        npt.assert_raises(TypeError, svd.svd_thresh, 1)

        npt.assert_raises(ValueError, svd.svd_thresh, self.data1, n_pc='bla')

    def test_svd_thresh_coef(self):
        """Test svd_thresh_coef."""
        npt.assert_almost_equal(
            svd.svd_thresh_coef(self.data1, lambda x_val: x_val, 0),
            self.data1,
            err_msg='Incorrect SVD coefficient tresholding',
        )

        npt.assert_raises(TypeError, svd.svd_thresh_coef, self.data1, 0, 0)


class ValidationTestCase(TestCase):
    """Test case for validation module."""

    def setUp(self):
        """Set test parameter values."""
        self.data1 = np.arange(9).reshape(3, 3).astype(float)

    def tearDown(self):
        """Unset test parameter values."""
        self.data1 = None

    def test_transpose_test(self):
        """Test transpose_test."""
        np.random.seed(2)
        npt.assert_equal(
            validation.transpose_test(
                lambda x_val, y_val: x_val.dot(y_val),
                lambda x_val, y_val: x_val.dot(y_val.T),
                self.data1.shape,
                x_args=self.data1,
            ),
            None,
        )

        npt.assert_raises(
            TypeError,
            validation.transpose_test,
            0,
            0,
            self.data1.shape,
            x_args=self.data1,
        )


class WaveletTestCase(TestCase):
    """Test case for wavelet module."""

    def setUp(self):
        """Set test parameter values."""
        self.data1 = np.arange(9).reshape(3, 3).astype(float)
        self.data2 = np.arange(36).reshape(4, 3, 3).astype(float)
        self.data3 = np.array([
            [
                [6.0, 20, 26.0],
                [36.0, 84.0, 84.0],
                [90, 164.0, 134.0],
            ],
            [
                [78.0, 155.0, 134.0],
                [225.0, 408.0, 327.0],
                [270, 461.0, 350],
            ],
            [
                [150, 290, 242.0],
                [414.0, 732.0, 570],
                [450, 758.0, 566.0],
            ],
            [
                [222.0, 425.0, 350],
                [603.0, 1056.0, 813.0],
                [630, 1055.0, 782.0],
            ],
        ])

        self.data4 = np.array([
            [6496.0, 9796.0, 6544.0],
            [9924.0, 14910, 9924.0],
            [6544.0, 9796.0, 6496.0],
        ])

        self.data5 = np.array([
            [[0, 1.0, 4.0], [3.0, 10, 13.0], [6.0, 19.0, 22.0]],
            [[3.0, 10, 13.0], [24.0, 46.0, 40], [45.0, 82.0, 67.0]],
            [[6.0, 19.0, 22.0], [45.0, 82.0, 67.0], [84.0, 145.0, 112.0]],
        ])

    def tearDown(self):
        """Unset test parameter values."""
        self.data1 = None
        self.data2 = None
        self.data3 = None
        self.data4 = None
        self.data5 = None

    def test_filter_convolve(self):
        """Test filter_convolve."""
        npt.assert_almost_equal(
            wavelet.filter_convolve(self.data1, self.data2),
            self.data3,
            err_msg='Inccorect filter comvolution.',
        )

        npt.assert_almost_equal(
            wavelet.filter_convolve(self.data2, self.data2, filter_rot=True),
            self.data4,
            err_msg='Inccorect filter comvolution.',
        )

    def test_filter_convolve_stack(self):
        """Test filter_convolve_stack."""
        npt.assert_almost_equal(
            wavelet.filter_convolve_stack(self.data1, self.data1),
            self.data5,
            err_msg='Inccorect filter stack comvolution.',
        )
