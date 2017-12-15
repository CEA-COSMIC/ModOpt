# -*- coding: utf-8 -*-

"""UNIT TESTS FOR SIGNAL

This module contains unit tests for the modopt.signal module.

:Author: Samuel Farrens <samuel.farrens@cea.fr>

"""

from unittest import TestCase
import numpy as np
import numpy.testing as npt
from modopt.opt import *


class ReweightTestCase(TestCase):

    def setUp(self):

        self.data1 = np.arange(9).reshape(3, 3).astype(float) + 1
        self.data2 = np.array([[0.5, 1., 1.5], [2., 2.5, 3.], [3.5, 4., 4.5]])
        self.rw = reweight.cwbReweight(self.data1)
        self.rw.reweight(self.data1)

    def tearDown(self):

        self.data1 = None
        self.data2 = None
        self.rw = None

    def test_cwbReweight(self):

        npt.assert_array_equal(self.rw.weights, self.data2,
                               err_msg='Incorrect CWB re-weighting.')

        npt.assert_raises(ValueError, self.rw.reweight, self.data1[0])
