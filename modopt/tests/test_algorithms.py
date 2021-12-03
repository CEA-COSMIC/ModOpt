# -*- coding: utf-8 -*-

"""UNIT TESTS FOR OPT.ALGORITHMS.

This module contains unit tests for the modopt.opt.algorithms module.

:Author: Samuel Farrens <samuel.farrens@cea.fr>

"""

from unittest import TestCase

import numpy as np
import numpy.testing as npt

from modopt.opt import algorithms, cost, gradient, linear, proximity, reweight

# Basic functions to be used as operators or as dummy functions
func_identity = lambda x_val: x_val
func_double = lambda x_val: x_val * 2
func_sq = lambda x_val: x_val ** 2
func_cube = lambda x_val: x_val ** 3


class Dummy(object):
    """Dummy class for tests."""

    pass


class AlgorithmTestCase(TestCase):
    """Test case for algorithms module."""

    def setUp(self):
        """Set test parameter values."""
        self.data1 = np.arange(9).reshape(3, 3).astype(float)
        self.data2 = self.data1 + np.random.randn(*self.data1.shape) * 1e-6
        self.data3 = np.arange(9).reshape(3, 3).astype(float) + 1

        grad_inst = gradient.GradBasic(
            self.data1,
            func_identity,
            func_identity,
        )

        prox_inst = proximity.Positivity()
        prox_dual_inst = proximity.IdentityProx()
        linear_inst = linear.Identity()
        reweight_inst = reweight.cwbReweight(self.data3)
        cost_inst = cost.costObj([grad_inst, prox_inst, prox_dual_inst])
        self.setup = algorithms.SetUp()
        self.max_iter = 20

        self.fb_all_iter = algorithms.ForwardBackward(
            self.data1,
            grad=grad_inst,
            prox=prox_inst,
            cost=None,
            auto_iterate=False,
            beta_update=func_identity,
        )
        self.fb_all_iter.iterate(self.max_iter)

        self.fb1 = algorithms.ForwardBackward(
            self.data1,
            grad=grad_inst,
            prox=prox_inst,
            beta_update=func_identity,
        )

        self.fb2 = algorithms.ForwardBackward(
            self.data1,
            grad=grad_inst,
            prox=prox_inst,
            cost=cost_inst,
            lambda_update=None,
        )

        self.fb3 = algorithms.ForwardBackward(
            self.data1,
            grad=grad_inst,
            prox=prox_inst,
            beta_update=func_identity,
            a_cd=3,
        )

        self.fb4 = algorithms.ForwardBackward(
            self.data1,
            grad=grad_inst,
            prox=prox_inst,
            beta_update=func_identity,
            r_lazy=3,
            p_lazy=0.7,
            q_lazy=0.7,
        )

        self.fb5 = algorithms.ForwardBackward(
            self.data1,
            grad=grad_inst,
            prox=prox_inst,
            restart_strategy='adaptive',
            xi_restart=0.9,
        )

        self.fb6 = algorithms.ForwardBackward(
            self.data1,
            grad=grad_inst,
            prox=prox_inst,
            restart_strategy='greedy',
            xi_restart=0.9,
            min_beta=1.0,
            s_greedy=1.1,
        )

        self.gfb_all_iter = algorithms.GenForwardBackward(
            self.data1,
            grad=grad_inst,
            prox_list=[prox_inst, prox_dual_inst],
            cost=None,
            auto_iterate=False,
            gamma_update=func_identity,
            beta_update=func_identity,
        )
        self.gfb_all_iter.iterate(self.max_iter)

        self.gfb1 = algorithms.GenForwardBackward(
            self.data1,
            grad=grad_inst,
            prox_list=[prox_inst, prox_dual_inst],
            gamma_update=func_identity,
            lambda_update=func_identity,
        )

        self.gfb2 = algorithms.GenForwardBackward(
            self.data1,
            grad=grad_inst,
            prox_list=[prox_inst, prox_dual_inst],
            cost=cost_inst,
        )

        self.gfb3 = algorithms.GenForwardBackward(
            self.data1,
            grad=grad_inst,
            prox_list=[prox_inst, prox_dual_inst],
            cost=cost_inst,
            step_size=2,
        )

        self.condat_all_iter = algorithms.Condat(
            self.data1,
            self.data2,
            grad=grad_inst,
            prox=prox_inst,
            cost=None,
            prox_dual=prox_dual_inst,
            sigma_update=func_identity,
            tau_update=func_identity,
            rho_update=func_identity,
            auto_iterate=False,
        )
        self.condat_all_iter.iterate(self.max_iter)

        self.condat1 = algorithms.Condat(
            self.data1,
            self.data2,
            grad=grad_inst,
            prox=prox_inst,
            prox_dual=prox_dual_inst,
            sigma_update=func_identity,
            tau_update=func_identity,
            rho_update=func_identity,
        )

        self.condat2 = algorithms.Condat(
            self.data1,
            self.data2,
            grad=grad_inst,
            prox=prox_inst,
            prox_dual=prox_dual_inst,
            linear=linear_inst,
            cost=cost_inst,
            reweight=reweight_inst,
        )

        self.condat3 = algorithms.Condat(
            self.data1,
            self.data2,
            grad=grad_inst,
            prox=prox_inst,
            prox_dual=prox_dual_inst,
            linear=Dummy(),
            cost=cost_inst,
            auto_iterate=False,
        )

        self.pogm_all_iter = algorithms.POGM(
            u=self.data1,
            x=self.data1,
            y=self.data1,
            z=self.data1,
            grad=grad_inst,
            prox=prox_inst,
            auto_iterate=False,
            cost=None,
        )
        self.pogm_all_iter.iterate(self.max_iter)

        self.pogm1 = algorithms.POGM(
            u=self.data1,
            x=self.data1,
            y=self.data1,
            z=self.data1,
            grad=grad_inst,
            prox=prox_inst,
        )

        self.vanilla_grad = algorithms.VanillaGenericGradOpt(
            self.data1,
            grad=grad_inst,
            prox=prox_inst,
            cost=cost_inst,
        )
        self.ada_grad = algorithms.AdaGenericGradOpt(
            self.data1,
            grad=grad_inst,
            prox=prox_inst,
            cost=cost_inst,
        )
        self.adam_grad = algorithms.ADAMGradOpt(
            self.data1,
            grad=grad_inst,
            prox=prox_inst,
            cost=cost_inst,
        )
        self.momentum_grad = algorithms.MomentumGradOpt(
            self.data1,
            grad=grad_inst,
            prox=prox_inst,
            cost=cost_inst,
        )
        self.rms_grad = algorithms.RMSpropGradOpt(
            self.data1,
            grad=grad_inst,
            prox=prox_inst,
            cost=cost_inst,
        )
        self.saga_grad = algorithms.SAGAOptGradOpt(
            self.data1,
            grad=grad_inst,
            prox=prox_inst,
            cost=cost_inst,
        )

        self.dummy = Dummy()
        self.dummy.cost = func_identity
        self.setup._check_operator(self.dummy.cost)

    def tearDown(self):
        """Unset test parameter values."""
        self.data1 = None
        self.data2 = None
        self.setup = None
        self.fb_all_iter = None
        self.fb1 = None
        self.fb2 = None
        self.gfb_all_iter = None
        self.gfb1 = None
        self.gfb2 = None
        self.condat_all_iter = None
        self.condat1 = None
        self.condat2 = None
        self.condat3 = None
        self.pogm1 = None
        self.pogm_all_iter = None
        self.dummy = None

    def test_set_up(self):
        """Test set_up."""
        npt.assert_raises(TypeError, self.setup._check_input_data, 1)

        npt.assert_raises(TypeError, self.setup._check_param, 1)

        npt.assert_raises(TypeError, self.setup._check_param_update, 1)

    def test_all_iter(self):
        """Test if all opt run for all iterations."""
        opts = [
            self.fb_all_iter,
            self.gfb_all_iter,
            self.condat_all_iter,
            self.pogm_all_iter,
        ]
        for opt in opts:
            npt.assert_equal(opt.idx, self.max_iter - 1)

    def test_forward_backward(self):
        """Test forward_backward."""
        npt.assert_array_equal(
            self.fb1.x_final,
            self.data1,
            err_msg='Incorrect ForwardBackward result.',
        )

        npt.assert_array_equal(
            self.fb2.x_final,
            self.data1,
            err_msg='Incorrect ForwardBackward result.',
        )

        npt.assert_array_equal(
            self.fb3.x_final,
            self.data1,
            err_msg='Incorrect ForwardBackward result.',
        )

        npt.assert_array_equal(
            self.fb4.x_final,
            self.data1,
            err_msg='Incorrect ForwardBackward result.',
        )

        npt.assert_array_equal(
            self.fb5.x_final,
            self.data1,
            err_msg='Incorrect ForwardBackward result.',
        )

        npt.assert_array_equal(
            self.fb6.x_final,
            self.data1,
            err_msg='Incorrect ForwardBackward result.',
        )

    def test_gen_forward_backward(self):
        """Test gen_forward_backward."""
        npt.assert_array_equal(
            self.gfb1.x_final,
            self.data1,
            err_msg='Incorrect GenForwardBackward result.',
        )

        npt.assert_array_equal(
            self.gfb2.x_final,
            self.data1,
            err_msg='Incorrect GenForwardBackward result.',
        )

        npt.assert_array_equal(
            self.gfb3.x_final,
            self.data1,
            err_msg='Incorrect GenForwardBackward result.',
        )

        npt.assert_equal(
            self.gfb3.step_size,
            2,
            err_msg='Incorrect step size.',
        )

        npt.assert_raises(
            TypeError,
            algorithms.GenForwardBackward,
            self.data1,
            self.dummy,
            [self.dummy],
            weights=1,
        )

        npt.assert_raises(
            ValueError,
            algorithms.GenForwardBackward,
            self.data1,
            self.dummy,
            [self.dummy],
            weights=[1],
        )

        npt.assert_raises(
            ValueError,
            algorithms.GenForwardBackward,
            self.data1,
            self.dummy,
            [self.dummy],
            weights=[0.5, 0.5],
        )

        npt.assert_raises(
            ValueError,
            algorithms.GenForwardBackward,
            self.data1,
            self.dummy,
            [self.dummy],
            weights=[0.5],
        )

    def test_condat(self):
        """Test gen_condat."""
        npt.assert_almost_equal(
            self.condat1.x_final,
            self.data1,
            err_msg='Incorrect Condat result.',
        )

        npt.assert_almost_equal(
            self.condat2.x_final,
            self.data1,
            err_msg='Incorrect Condat result.',
        )

    def test_pogm(self):
        """Test pogm."""
        npt.assert_almost_equal(
            self.pogm1.x_final,
            self.data1,
            err_msg='Incorrect POGM result.',
        )

    def test_ada_grad(self):
        """Test ADA Gradient Descent."""
        self.ada_grad.iterate()
        npt.assert_almost_equal(
            self.ada_grad.x_final,
            self.data1,
            err_msg='Incorrect ADAGrad results.',
        )

    def test_adam_grad(self):
        """Test ADAM Gradient Descent."""
        self.adam_grad.iterate()
        npt.assert_almost_equal(
            self.adam_grad.x_final,
            self.data1,
            err_msg='Incorrect ADAMGrad results.',
        )

    def test_momemtum_grad(self):
        """Test Momemtum Gradient Descent."""
        self.momentum_grad.iterate()
        npt.assert_almost_equal(
            self.momentum_grad.x_final,
            self.data1,
            err_msg='Incorrect MomentumGrad results.',
        )

    def test_rmsprop_grad(self):
        """Test RMSProp Gradient Descent."""
        self.rms_grad.iterate()
        npt.assert_almost_equal(
            self.rms_grad.x_final,
            self.data1,
            err_msg='Incorrect RMSPropGrad results.',
        )

    def test_saga_grad(self):
        """Test SAGA Descent."""
        self.saga_grad.iterate()
        npt.assert_almost_equal(
            self.saga_grad.x_final,
            self.data1,
            err_msg='Incorrect SAGA Grad results.',
        )

    def test_vanilla_grad(self):
        """Test Vanilla Gradient Descent."""
        self.vanilla_grad.iterate()
        npt.assert_almost_equal(
            self.vanilla_grad.x_final,
            self.data1,
            err_msg='Incorrect VanillaGrad results.',
        )
