# -*- coding: utf-8 -*-

"""UNIT TESTS FOR SIGNAL

This module contains unit tests for the modopt.signal module.

:Author: Samuel Farrens <samuel.farrens@cea.fr>

"""

from unittest import TestCase
import numpy as np
import numpy.testing as npt
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

    def tearDown(self):

        self.parent = None
        self.ident = None
        self.combo = None
        self.combo_weight = None
        self.data1 = None

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
