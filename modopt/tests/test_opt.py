# -*- coding: utf-8 -*-

"""UNIT TESTS FOR SIGNAL

This module contains unit tests for the modopt.signal module.

:Author: Samuel Farrens <samuel.farrens@cea.fr>

"""

from unittest import TestCase
import numpy as np
import numpy.testing as npt
from builtins import zip
from modopt.opt import *


class CostTestCase(TestCase):

    def setUp(self):

        class dummy(object):
            pass

        dummy_inst1 = dummy()
        dummy_inst1.cost = lambda x: x ** 2
        dummy_inst2 = dummy()
        dummy_inst2.cost = lambda x: x ** 3

        self.inst1 = cost.costObj([dummy_inst1, dummy_inst2])
        [self.inst1.get_cost(2) for i in range(2)]
        self.inst2 = cost.costObj([dummy_inst1, dummy_inst2], cost_interval=2)
        [self.inst2.get_cost(2) for i in range(6)]
        self.dummy = dummy()

    def tearDown(self):

        self.inst = None

    def test_cost_object(self):

        npt.assert_equal(self.inst1.get_cost(2), False,
                         err_msg='Incorrect cost test result.')

        npt.assert_equal(self.inst1.get_cost(2), True,
                         err_msg='Incorrect cost test result.')

        npt.assert_equal(self.inst1.cost, 12, err_msg='Incorrect cost value.')

        npt.assert_equal(self.inst2.cost, 12, err_msg='Incorrect cost value.')

        npt.assert_raises(TypeError, cost.costObj, 1)

        npt.assert_raises(ValueError, cost.costObj, [self.dummy, self.dummy])


class GradientTestCase(TestCase):

    def setUp(self):

        self.data1 = np.arange(9).reshape(3, 3).astype(float)
        self.gb = gradient.GradParent(self.data1, lambda x: x ** 2,
                                      lambda x: x ** 3)
        self.gb.get_grad(self.data1)

    def tearDown(self):

        self.data1 = None
        self.gb = None

    def test_grad_basic_operators(self):

        npt.assert_array_equal(self.gb.MX(self.data1), np.array([[0., 1., 4.],
                               [9., 16., 25.], [36., 49., 64.]]),
                               err_msg="Incorrect gradient: MX.")

        npt.assert_array_equal(self.gb.MtX(self.data1), np.array([[0., 1., 8.],
                               [27., 64., 125.], [216., 343., 512.]]),
                               err_msg="Incorrect gradient: MtX.")

        npt.assert_array_equal(self.gb.MtMX(self.data1),
                               np.array([[0.00000000e+00, 1.00000000e+00,
                                          6.40000000e+01],
                                         [7.29000000e+02, 4.09600000e+03,
                                          1.56250000e+04],
                                         [4.66560000e+04, 1.17649000e+05,
                                          2.62144000e+05]]),
                               err_msg="Incorrect gradient: MtMX.")

    def test_grad_basic_gradient(self):

        npt.assert_array_equal(self.gb.grad,
                               np.array([[0.00000000e+00, 0.00000000e+00,
                                          8.00000000e+00],
                                         [2.16000000e+02, 1.72800000e+03,
                                          8.00000000e+03],
                                         [2.70000000e+04, 7.40880000e+04,
                                          1.75616000e+05]]),
                               err_msg="Incorrect gradient.")

        npt.assert_raises(TypeError, gradient.GradParent, 1,
                          lambda x: x ** 2, lambda x: x ** 3)


class LinearTestCase(TestCase):

    def setUp(self):

        self.parent = linear.LinearParent(lambda x: x ** 2, lambda x: x ** 3)
        self.ident = linear.Identity()
        self.combo = linear.LinearCombo([self.parent, self.parent])
        self.combo_weight = linear.LinearCombo([self.parent, self.parent],
                                               [1.0, 1.0])
        self.data1 = np.arange(18).reshape(2, 3, 3).astype(float)

        class dummy(object):
            pass

        self.dummy = dummy()

    def tearDown(self):

        self.parent = None
        self.ident = None
        self.combo = None
        self.combo_weight = None
        self.data1 = None
        self.dummy = None

    def test_linear_parent(self):

        npt.assert_equal(self.parent.op(2), 4, err_msg='Incorrect linear '
                                                       'parent operation.')

        npt.assert_equal(self.parent.adj_op(2), 8, err_msg='Incorrect linear '
                                                           'parent adjoint '
                                                           'operation.')

        npt.assert_raises(TypeError, linear.LinearParent, 0, 0)

    def test_identity(self):

        npt.assert_equal(self.ident.op(1.0), 1.0,
                         err_msg='Incorrect identity operation.')

        npt.assert_equal(self.ident.adj_op(1.0), 1.0,
                         err_msg='Incorrect identity adjoint operation.')

    def test_linear_combo(self):

        npt.assert_equal(self.combo.op(2), np.array([4, 4]).astype(object),
                         err_msg='Incorrect combined linear operation')

        npt.assert_equal(self.combo.adj_op([2, 2]), 8.0,
                         err_msg='Incorrect combined linear adjoint operation')

        npt.assert_raises(TypeError, linear.LinearCombo, self.parent)

        npt.assert_raises(ValueError, linear.LinearCombo, [])

        npt.assert_raises(ValueError, linear.LinearCombo, [self.dummy])

        self.dummy.op = lambda x: x

        npt.assert_raises(ValueError, linear.LinearCombo, [self.dummy])

    def test_linear_combo_weight(self):

        npt.assert_equal(self.combo_weight.op(2),
                         np.array([4, 4]).astype(object),
                         err_msg='Incorrect combined linear operation')

        npt.assert_equal(self.combo_weight.adj_op([2, 2]), 16.0,
                         err_msg='Incorrect combined linear adjoint operation')

        npt.assert_raises(ValueError, linear.LinearCombo,
                          [self.parent, self.parent], [1.0])

        npt.assert_raises(TypeError, linear.LinearCombo,
                          [self.parent, self.parent], ['1', '1'])


class ProximityTestCase(TestCase):

    def setUp(self):

        self.parent = proximity.ProximityParent(lambda x: x ** 2,
                                                lambda x: x * 2)
        self.identity = proximity.IdentityProx()
        self.positivity = proximity.Positivity()
        weights = np.ones(9).reshape(3, 3).astype(float) * 3
        self.sparsethresh = proximity.SparseThreshold(linear.Identity(),
                                                      weights)
        self.lowrank = proximity.LowRankMatrix(10.0, thresh_type='hard')
        self.lowrank_ngole = proximity.LowRankMatrix(10.0, lowr_type='ngole',
                                                     operator=lambda x: x * 2)
        self.combo = proximity.ProximityCombo([self.identity, self.positivity])
        self.data1 = np.arange(9).reshape(3, 3).astype(float)
        self.data2 = np.array([[-0., -0., -0.], [0., 1., 2.], [3., 4., 5.]])
        self.data3 = np.arange(18).reshape(2, 3, 3).astype(float)
        self.data4 = np.array([[[2.73843189, 3.14594066, 3.55344943],
                                [3.9609582, 4.36846698, 4.77597575],
                                [5.18348452, 5.59099329, 5.99850206]],
                               [[8.07085295, 9.2718846, 10.47291625],
                                [11.67394789, 12.87497954, 14.07601119],
                                [15.27704284, 16.47807449, 17.67910614]]])
        self.data5 = np.array([[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                               [[4.00795282, 4.60438026, 5.2008077],
                                [5.79723515, 6.39366259, 6.99009003],
                                [7.58651747, 8.18294492, 8.77937236]]])
        self.data6 = self.data3 * -1
        self.data7 = self.combo.op(self.data6)
        self.data8 = np.empty(2, dtype=np.ndarray)
        self.data8[0] = np.array([[-0., -1., -2.], [-3., -4., -5.],
                                  [-6., -7., -8.]])
        self.data8[1] = np.array([[-0., -0., -0.], [-0., -0., -0.],
                                  [-0., -0., -0.]])

        class dummy(object):
            pass

        self.dummy = dummy()

    def tearDown(self):

        self.parent = None
        self.identity = None
        self.positivity = None
        self.sparsethresh = None
        self.lowrank = None
        self.combo = None
        self.data1 = None
        self.data2 = None
        self.data3 = None
        self.data4 = None
        self.data5 = None
        self.data6 = None
        self.data7 = None
        self.data8 = None
        self.dummy = None

    def test_proximity_parent(self):

        npt.assert_equal(self.parent.op(3), 9,
                         err_msg='Inccoret proximity parent operation.')

        npt.assert_equal(self.parent.cost(3), 6,
                         err_msg='Inccoret proximity parent cost.')

    def test_identity(self):

        npt.assert_equal(self.identity.op(3), 3,
                         err_msg='Inccoret proximity identity operation.')

        npt.assert_equal(self.identity.cost(3), 0.0,
                         err_msg='Inccoret proximity identity cost.')

    def test_positivity(self):

        npt.assert_equal(self.positivity.op(-3), 0,
                         err_msg='Inccoret proximity positivity operation.')

        npt.assert_equal(self.positivity.cost(-3, verbose=True), 0.0,
                         err_msg='Inccoret proximity positivity cost.')

    def test_sparse_threshold(self):

        npt.assert_array_equal(self.sparsethresh.op(self.data1), self.data2,
                               err_msg='Inccorect sparse threshold operation.')

        npt.assert_equal(self.sparsethresh.cost(self.data1, verbose=True),
                         108.0, err_msg='Inccoret sparse threshold cost.')

    def test_low_rank_matrix(self):

        npt.assert_almost_equal(self.lowrank.op(self.data3), self.data4,
                                err_msg='Inccorect low rank operation: '
                                        'standard')

        npt.assert_almost_equal(self.lowrank_ngole.op(self.data3), self.data5,
                                err_msg='Inccorect low rank operation: '
                                        'ngole')

        npt.assert_almost_equal(self.lowrank.cost(self.data3, verbose=True),
                                469.39132942464983,
                                err_msg='Inccoret low rank cost.')

    def test_proximity_combo(self):

        for data7, data8 in zip(self.data7, self.data8):
            npt.assert_array_equal(data7, data8,
                                   err_msg='Inccorect combined operation')

        npt.assert_equal(self.combo.cost(self.data6), 0.0,
                         err_msg='Inccoret combined cost.')

        npt.assert_raises(TypeError, proximity.ProximityCombo, 1)

        npt.assert_raises(ValueError, proximity.ProximityCombo, [])

        npt.assert_raises(ValueError, proximity.ProximityCombo, [self.dummy])

        self.dummy.op = lambda x: x

        npt.assert_raises(ValueError, proximity.ProximityCombo, [self.dummy])


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
