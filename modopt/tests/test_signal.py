# -*- coding: utf-8 -*-

"""UNIT TESTS FOR SIGNAL

This module contains unit tests for the modopt.signal module.

:Author: Samuel Farrens <samuel.farrens@cea.fr>

"""

from unittest import TestCase
import numpy as np
import numpy.testing as npt
from modopt.signal import *


class NoiseTestCase(TestCase):

        def setUp(self):

            self.data1 = np.arange(9).reshape(3, 3).astype(float)
            self.data2 = np.array([[0., 2., 2.], [4., 5., 10.],
                                   [11., 15., 18.]])
            self.data3 = np.array([[1.62434536, 0.38824359, 1.47182825],
                                   [1.92703138, 4.86540763, 2.6984613],
                                   [7.74481176, 6.2387931, 8.3190391]])
            self.data4 = np.array([[0., 0., 0.], [0., 0., 5.], [6., 7., 8.]])
            self.data5 = np.array([[-0., -0., -0.], [-0., -0., 0.],
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
