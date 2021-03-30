# -*- coding: utf-8 -*-

"""UNIT TESTS FOR SIGNAL.

This module contains unit tests for the modopt.signal module.

:Author: Samuel Farrens <samuel.farrens@cea.fr>

"""

from builtins import zip
from unittest import TestCase, skipIf, skipUnless

import numpy as np
import numpy.testing as npt

from modopt.opt import algorithms, cost, gradient, linear, proximity, reweight

try:
    import sklearn
except ImportError:  # pragma: no cover
    import_sklearn = False
else:
    import_sklearn = True


dummy_func = lambda x_val: x_val
dummy_func_double = lambda x_val: x_val * 2
dummy_func_sq = lambda x_val: x_val ** 2
dummy_func_cube = lambda x_val: x_val ** 3


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
            dummy_func,
            dummy_func,
        )

        prox_inst = proximity.Positivity()
        prox_dual_inst = proximity.IdentityProx()
        linear_inst = linear.Identity()
        reweight_inst = reweight.cwbReweight(self.data3)
        cost_inst = cost.costObj([grad_inst, prox_inst, prox_dual_inst])
        self.setup = algorithms.SetUp()

        self.fb1 = algorithms.ForwardBackward(
            self.data1,
            grad=grad_inst,
            prox=prox_inst,
            beta_update=dummy_func,
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
            beta_update=dummy_func,
            a_cd=3,
        )

        self.fb4 = algorithms.ForwardBackward(
            self.data1,
            grad=grad_inst,
            prox=prox_inst,
            beta_update=dummy_func,
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

        self.gfb1 = algorithms.GenForwardBackward(
            self.data1,
            grad=grad_inst,
            prox_list=[prox_inst, prox_dual_inst],
            gamma_update=dummy_func,
            lambda_update=dummy_func,
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

        self.condat1 = algorithms.Condat(
            self.data1,
            self.data2,
            grad=grad_inst,
            prox=prox_inst,
            prox_dual=prox_dual_inst,
            sigma_update=dummy_func,
            tau_update=dummy_func,
            rho_update=dummy_func,
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

        self.pogm1 = algorithms.POGM(
            u=self.data1,
            x=self.data1,
            y=self.data1,
            z=self.data1,
            grad=grad_inst,
            prox=prox_inst,
        )

        self.dummy = Dummy()
        self.dummy.cost = dummy_func
        self.setup._check_operator(self.dummy.cost)

    def tearDown(self):
        """Unset test parameter values."""
        self.data1 = None
        self.data2 = None
        self.setup = None
        self.fb1 = None
        self.fb2 = None
        self.gfb1 = None
        self.gfb2 = None
        self.condat1 = None
        self.condat2 = None
        self.condat3 = None
        self.dummy = None

    def test_set_up(self):
        """Test set_up."""
        npt.assert_raises(TypeError, self.setup._check_input_data, 1)

        npt.assert_raises(TypeError, self.setup._check_param, 1)

        npt.assert_raises(TypeError, self.setup._check_param_update, 1)

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


class CostTestCase(TestCase):
    """Test case for cost module."""

    def setUp(self):
        """Set test parameter values."""
        dummy_inst1 = Dummy()
        dummy_inst1.cost = dummy_func_sq
        dummy_inst2 = Dummy()
        dummy_inst2.cost = dummy_func_cube

        self.inst1 = cost.costObj([dummy_inst1, dummy_inst2])
        self.inst2 = cost.costObj([dummy_inst1, dummy_inst2], cost_interval=2)
        # Test that by default cost of False if interval is None
        self.inst_none = cost.costObj(
            [dummy_inst1, dummy_inst2],
            cost_interval=None,
        )
        for _ in range(2):
            self.inst1.get_cost(2)
        for _ in range(6):
            self.inst2.get_cost(2)
            self.inst_none.get_cost(2)
        self.dummy = Dummy()

    def tearDown(self):
        """Unset test parameter values."""
        self.inst = None

    def test_cost_object(self):
        """Test cost_object."""
        npt.assert_equal(
            self.inst1.get_cost(2),
            False,
            err_msg='Incorrect cost test result.',
        )
        npt.assert_equal(
            self.inst1.get_cost(2),
            True,
            err_msg='Incorrect cost test result.',
        )
        npt.assert_equal(
            self.inst_none.get_cost(2),
            False,
            err_msg='Incorrect cost test result.',
        )

        npt.assert_equal(self.inst1.cost, 12, err_msg='Incorrect cost value.')

        npt.assert_equal(self.inst2.cost, 12, err_msg='Incorrect cost value.')

        npt.assert_raises(TypeError, cost.costObj, 1)

        npt.assert_raises(ValueError, cost.costObj, [self.dummy, self.dummy])


class GradientTestCase(TestCase):
    """Test case for gradient module."""

    def setUp(self):
        """Set test parameter values."""
        self.data1 = np.arange(9).reshape(3, 3).astype(float)
        self.gp = gradient.GradParent(
            self.data1,
            dummy_func_sq,
            dummy_func_cube,
            dummy_func,
            lambda input_val: 1.0,
            data_type=np.floating,
        )
        self.gp.grad = self.gp.get_grad(self.data1)
        self.gb = gradient.GradBasic(
            self.data1,
            dummy_func_sq,
            dummy_func_cube,
        )
        self.gb.get_grad(self.data1)

    def tearDown(self):
        """Unset test parameter values."""
        self.data1 = None
        self.gp = None
        self.gb = None

    def test_grad_parent_operators(self):
        """Test GradParent."""
        npt.assert_array_equal(
            self.gp.op(self.data1),
            np.array([[0, 1.0, 4.0], [9.0, 16.0, 25.0], [36.0, 49.0, 64.0]]),
            err_msg='Incorrect gradient operation.',
        )

        npt.assert_array_equal(
            self.gp.trans_op(self.data1),
            np.array(
                [[0, 1.0, 8.0], [27.0, 64.0, 125.0], [216.0, 343.0, 512.0]],
            ),
            err_msg='Incorrect gradient transpose operation.',
        )

        npt.assert_array_equal(
            self.gp.trans_op_op(self.data1),
            np.array([
                [0, 1.0, 6.40000000e1],
                [7.29000000e2, 4.09600000e3, 1.56250000e4],
                [4.66560000e4, 1.17649000e5, 2.62144000e5],
            ]),
            err_msg='Incorrect gradient transpose operation operation.',
        )

        npt.assert_equal(
            self.gp.cost(self.data1),
            1.0,
            err_msg='Incorrect cost.',
        )

        npt.assert_raises(
            TypeError,
            gradient.GradParent,
            1,
            dummy_func_sq,
            dummy_func_cube,
        )

    def test_grad_basic_gradient(self):
        """Test GradBasic."""
        npt.assert_array_equal(
            self.gb.grad,
            np.array([
                [0, 0, 8.0],
                [2.16000000e2, 1.72800000e3, 8.0e3],
                [2.70000000e4, 7.40880000e4, 1.75616000e5],
            ]),
            err_msg='Incorrect gradient.',
        )


class LinearTestCase(TestCase):
    """Test case for linear module."""

    def setUp(self):
        """Set test parameter values."""
        self.parent = linear.LinearParent(
            dummy_func_sq,
            dummy_func_cube,
        )
        self.ident = linear.Identity()
        filters = np.arange(8).reshape(2, 2, 2).astype(float)
        self.wave = linear.WaveletConvolve(filters)
        self.combo = linear.LinearCombo([self.parent, self.parent])
        self.combo_weight = linear.LinearCombo(
            [self.parent, self.parent],
            [1.0, 1.0],
        )
        self.data1 = np.arange(18).reshape(2, 3, 3).astype(float)
        self.data2 = np.arange(4).reshape(1, 2, 2).astype(float)
        self.data3 = np.arange(8).reshape(1, 2, 2, 2).astype(float)
        self.data4 = np.array([[[[0, 0], [0, 4.0]], [[0, 4.0], [8.0, 28.0]]]])
        self.data5 = np.array([[[28.0, 62.0], [68.0, 140.0]]])
        self.dummy = Dummy()

    def tearDown(self):
        """Unset test parameter values."""
        self.parent = None
        self.ident = None
        self.combo = None
        self.combo_weight = None
        self.data1 = None
        self.data2 = None
        self.data3 = None
        self.data4 = None
        self.data5 = None
        self.dummy = None

    def test_linear_parent(self):
        """Test LinearParent."""
        npt.assert_equal(
            self.parent.op(2),
            4,
            err_msg='Incorrect linear parent operation.',
        )

        npt.assert_equal(
            self.parent.adj_op(2),
            8,
            err_msg='Incorrect linear parent adjoint operation.',
        )

        npt.assert_raises(TypeError, linear.LinearParent, 0, 0)

    def test_identity(self):
        """Test Identity."""
        npt.assert_equal(
            self.ident.op(1.0),
            1.0,
            err_msg='Incorrect identity operation.',
        )

        npt.assert_equal(
            self.ident.adj_op(1.0),
            1.0,
            err_msg='Incorrect identity adjoint operation.',
        )

    def test_wavelet_convolve(self):
        """Test WaveletConvolve."""
        npt.assert_almost_equal(
            self.wave.op(self.data2),
            self.data4,
            err_msg='Incorrect wavelet convolution operation.',
        )

        npt.assert_almost_equal(
            self.wave.adj_op(self.data3),
            self.data5,
            err_msg='Incorrect wavelet convolution adjoint operation.',
        )

    def test_linear_combo(self):
        """Test LinearCombo."""
        npt.assert_equal(
            self.combo.op(2),
            np.array([4, 4]).astype(object),
            err_msg='Incorrect combined linear operation',
        )

        npt.assert_equal(
            self.combo.adj_op([2, 2]),
            8.0,
            err_msg='Incorrect combined linear adjoint operation',
        )

        npt.assert_raises(TypeError, linear.LinearCombo, self.parent)

        npt.assert_raises(ValueError, linear.LinearCombo, [])

        npt.assert_raises(ValueError, linear.LinearCombo, [self.dummy])

        self.dummy.op = dummy_func

        npt.assert_raises(ValueError, linear.LinearCombo, [self.dummy])

    def test_linear_combo_weight(self):
        """Test LinearCombo with weight ."""
        npt.assert_equal(
            self.combo_weight.op(2),
            np.array([4, 4]).astype(object),
            err_msg='Incorrect combined linear operation',
        )

        npt.assert_equal(
            self.combo_weight.adj_op([2, 2]),
            16.0,
            err_msg='Incorrect combined linear adjoint operation',
        )

        npt.assert_raises(
            ValueError,
            linear.LinearCombo,
            [self.parent, self.parent],
            [1.0],
        )

        npt.assert_raises(
            TypeError,
            linear.LinearCombo,
            [self.parent, self.parent],
            ['1', '1'],
        )


class ProximityTestCase(TestCase):
    """Test case for proximity module."""

    def setUp(self):
        """Set test parameter values."""
        self.parent = proximity.ProximityParent(
            dummy_func_sq,
            dummy_func_double,
        )
        self.identity = proximity.IdentityProx()
        self.positivity = proximity.Positivity()
        weights = np.ones(9).reshape(3, 3).astype(float) * 3
        self.sparsethresh = proximity.SparseThreshold(
            linear.Identity(),
            weights,
        )
        self.lowrank = proximity.LowRankMatrix(10.0, thresh_type='hard')
        self.lowrank_ngole = proximity.LowRankMatrix(
            10.0,
            lowr_type='ngole',
            operator=dummy_func_double,
        )
        self.linear_comp = proximity.LinearCompositionProx(
            linear_op=linear.Identity(),
            prox_op=self.sparsethresh,
        )
        self.combo = proximity.ProximityCombo([self.identity, self.positivity])
        if import_sklearn:
            self.owl = proximity.OrderedWeightedL1Norm(weights.flatten())
        self.ridge = proximity.Ridge(linear.Identity(), weights)
        self.elasticnet_alpha0 = proximity.ElasticNet(
            linear.Identity(),
            alpha=0,
            beta=weights,
        )
        self.elasticnet_beta0 = proximity.ElasticNet(
            linear.Identity(),
            alpha=weights,
            beta=0,
        )
        self.one_support = proximity.KSupportNorm(beta=0.2, k_value=1)
        self.five_support_norm = proximity.KSupportNorm(beta=3, k_value=5)
        self.d_support = proximity.KSupportNorm(beta=3.0 * 2, k_value=19)
        self.group_lasso = proximity.GroupLASSO(
            weights=np.tile(weights, (4, 1, 1)),
        )
        self.data1 = np.arange(9).reshape(3, 3).astype(float)
        self.data2 = np.array([[-0, -0, -0], [0, 1.0, 2.0], [3.0, 4.0, 5.0]])
        self.data3 = np.arange(18).reshape(2, 3, 3).astype(float)
        self.data4 = np.array([
            [
                [2.73843189, 3.14594066, 3.55344943],
                [3.9609582, 4.36846698, 4.77597575],
                [5.18348452, 5.59099329, 5.99850206],
            ],
            [
                [8.07085295, 9.2718846, 10.47291625],
                [11.67394789, 12.87497954, 14.07601119],
                [15.27704284, 16.47807449, 17.67910614],
            ],
        ])
        self.data5 = np.array([
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [
                [4.00795282, 4.60438026, 5.2008077],
                [5.79723515, 6.39366259, 6.99009003],
                [7.58651747, 8.18294492, 8.77937236],
            ],
        ])
        self.data6 = self.data3 * -1
        self.data7 = self.combo.op(self.data6)
        self.data8 = np.empty(2, dtype=np.ndarray)
        self.data8[0] = np.array(
            [[-0, -1.0, -2.0], [-3.0, -4.0, -5.0], [-6.0, -7.0, -8.0]],
        )
        self.data8[1] = np.array(
            [[-0, -0, -0], [-0, -0, -0], [-0, -0, -0]],
        )
        self.data9 = self.data1 * (1 + 1j)
        self.data10 = self.data9 / (2 * 3 + 1)
        self.data11 = np.asarray(
            [[0, 0, 0], [0, 1.0, 1.25], [1.5, 1.75, 2.0]],
        )
        self.random_data = 3 * np.random.random(
            self.group_lasso.weights[0].shape,
        )
        self.random_data_tile = np.tile(
            self.random_data,
            (self.group_lasso.weights.shape[0], 1, 1),
        )
        self.gl_result_data = 2 * self.random_data_tile - 3
        self.gl_result_data = np.array(
            (self.gl_result_data * (self.gl_result_data > 0).astype('int'))
            / 2,
        )

        self.dummy = Dummy()

    def tearDown(self):
        """Unset test parameter values."""
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
        self.random_data = None
        self.random_data_tile = None
        self.gl_result_data = None

    def test_proximity_parent(self):
        """Test ProximityParent."""
        npt.assert_equal(
            self.parent.op(3),
            9,
            err_msg='Inccoret proximity parent operation.',
        )

        npt.assert_equal(
            self.parent.cost(3),
            6,
            err_msg='Inccoret proximity parent cost.',
        )

    def test_identity(self):
        """Test IdentityProx."""
        npt.assert_equal(
            self.identity.op(3),
            3,
            err_msg='Inccoret proximity identity operation.',
        )

        npt.assert_equal(
            self.identity.cost(3),
            0,
            err_msg='Inccoret proximity identity cost.',
        )

    def test_positivity(self):
        """Test Positivity."""
        npt.assert_equal(
            self.positivity.op(-3),
            0,
            err_msg='Inccoret proximity positivity operation.',
        )

        npt.assert_equal(
            self.positivity.cost(-3, verbose=True),
            0,
            err_msg='Inccoret proximity positivity cost.',
        )

    def test_sparse_threshold(self):
        """Test SparseThreshold."""
        npt.assert_array_equal(
            self.sparsethresh.op(self.data1),
            self.data2,
            err_msg='Inccorect sparse threshold operation.',
        )

        npt.assert_equal(
            self.sparsethresh.cost(self.data1, verbose=True),
            108.0,
            err_msg='Inccoret sparse threshold cost.',
        )

    def test_low_rank_matrix(self):
        """Test LowRankMatrix."""
        npt.assert_almost_equal(
            self.lowrank.op(self.data3),
            self.data4,
            err_msg='Inccorect low rank operation: standard',
        )

        npt.assert_almost_equal(
            self.lowrank_ngole.op(self.data3),
            self.data5,
            err_msg='Inccorect low rank operation: ngole',
        )

        npt.assert_almost_equal(
            self.lowrank.cost(self.data3, verbose=True),
            469.39132942464983,
            err_msg='Inccoret low rank cost.',
        )

    def test_linear_comp_prox(self):
        """Test LinearCompositionProx."""
        npt.assert_array_equal(
            self.linear_comp.op(self.data1),
            self.data2,
            err_msg='Inccorect sparse threshold operation.',
        )

        npt.assert_equal(
            self.linear_comp.cost(self.data1, verbose=True),
            108.0,
            err_msg='Inccoret sparse threshold cost.',
        )

    def test_proximity_combo(self):
        """Test ProximityCombo."""
        for data7, data8 in zip(self.data7, self.data8):
            npt.assert_array_equal(
                data7,
                data8,
                err_msg='Inccorect combined operation',
            )

        npt.assert_equal(
            self.combo.cost(self.data6),
            0,
            err_msg='Incorrect combined cost.',
        )

        npt.assert_raises(TypeError, proximity.ProximityCombo, 1)

        npt.assert_raises(ValueError, proximity.ProximityCombo, [])

        npt.assert_raises(ValueError, proximity.ProximityCombo, [self.dummy])

        self.dummy.op = dummy_func

        npt.assert_raises(ValueError, proximity.ProximityCombo, [self.dummy])

    @skipIf(import_sklearn, 'sklearn is installed.')  # pragma: no cover
    def test_owl_sklearn_error(self):
        """Test OrderedWeightedL1Norm with Scikit-Learn."""
        npt.assert_raises(ImportError, proximity.OrderedWeightedL1Norm, 1)

    @skipUnless(import_sklearn, 'sklearn not installed.')  # pragma: no cover
    def test_sparse_owl(self):
        """Test OrderedWeightedL1Norm."""
        npt.assert_array_equal(
            self.owl.op(self.data1.flatten()),
            self.data2.flatten(),
            err_msg='Incorrect sparse threshold operation.',
        )

        npt.assert_equal(
            self.owl.cost(self.data1.flatten(), verbose=True),
            108.0,
            err_msg='Incorrect sparse threshold cost.',
        )

        npt.assert_raises(
            ValueError,
            proximity.OrderedWeightedL1Norm,
            np.arange(10),
        )

    def test_ridge(self):
        """Test Ridge."""
        npt.assert_array_equal(
            self.ridge.op(self.data9),
            self.data10,
            err_msg='Incorect shrinkage operation.',
        )

        npt.assert_equal(
            self.ridge.cost(self.data9, verbose=True),
            408.0 * 3.0,
            err_msg='Incorect shrinkage cost.',
        )

    def test_elastic_net_alpha0(self):
        """Test ElasticNet."""
        npt.assert_array_equal(
            self.elasticnet_alpha0.op(self.data1),
            self.data2,
            err_msg='Incorect sparse threshold operation ElasticNet class.',
        )

        npt.assert_equal(
            self.elasticnet_alpha0.cost(self.data1),
            108.0,
            err_msg='Incorect shrinkage cost in ElasticNet class.',
        )

    def test_elastic_net_beta0(self):
        """Test ElasticNet with beta=0."""
        npt.assert_array_equal(
            self.elasticnet_beta0.op(self.data9),
            self.data10,
            err_msg='Incorect ridge operation ElasticNet class.',
        )

        npt.assert_equal(
            self.elasticnet_beta0.cost(self.data9, verbose=True),
            408.0 * 3.0,
            err_msg='Incorect shrinkage cost in ElasticNet class.',
        )

    def test_one_support_norm(self):
        """Test KSupportNorm with k=1."""
        npt.assert_allclose(
            self.one_support.op(self.data1.flatten()),
            self.data2.flatten(),
            err_msg='Incorect sparse threshold operation for 1-support norm',
            rtol=1e-6,
        )

        npt.assert_equal(
            self.one_support.cost(self.data1.flatten(), verbose=True),
            259.2,
            err_msg='Incorect sparse threshold cost.',
        )

        npt.assert_raises(ValueError, proximity.KSupportNorm, 0, 0)

    def test_three_support_norm(self):
        """Test KSupportNorm with k=5."""
        npt.assert_allclose(
            self.five_support_norm.op(self.data1.flatten()),
            self.data11.flatten(),
            err_msg='Incorect sparse Ksupport norm operation',
            rtol=1e-6,
        )

        npt.assert_equal(
            self.five_support_norm.cost(self.data1.flatten(), verbose=True),
            684.0,
            err_msg='Inccoret 3-support norm cost.',
        )

        npt.assert_raises(ValueError, proximity.KSupportNorm, 0, 0)

    def test_d_support_norm(self):
        """Test KSupportNorm with k=19."""
        npt.assert_allclose(
            self.d_support.op(self.data9.flatten()),
            self.data10.flatten(),
            err_msg='Incorect shrinkage operation for d-support norm',
            rtol=1e-6,
        )

        npt.assert_almost_equal(
            self.d_support.cost(self.data9.flatten(), verbose=True),
            408.0 * 3.0,
            err_msg='Incorect shrinkage cost for d-support norm.',
        )

        npt.assert_raises(ValueError, proximity.KSupportNorm, 0, 0)

    def test_group_lasso(self):
        """Test GroupLASSO."""
        npt.assert_allclose(
            self.group_lasso.op(self.random_data_tile),
            self.gl_result_data,
        )
        npt.assert_equal(
            self.group_lasso.cost(self.random_data_tile),
            np.sum(6 * self.random_data_tile),
        )
        # Check that for 0 weights operator doesnt change result
        self.group_lasso.weights = np.zeros_like(self.group_lasso.weights)
        npt.assert_equal(
            self.group_lasso.op(self.random_data_tile),
            self.random_data_tile,
        )
        npt.assert_equal(self.group_lasso.cost(self.random_data_tile), 0)


class ReweightTestCase(TestCase):
    """Test case for reweight module."""

    def setUp(self):
        """Set test parameter values."""
        self.data1 = np.arange(9).reshape(3, 3).astype(float) + 1
        self.data2 = np.array(
            [[0.5, 1.0, 1.5], [2.0, 2.5, 3.0], [3.5, 4.0, 4.5]],
        )
        self.rw = reweight.cwbReweight(self.data1)
        self.rw.reweight(self.data1)

    def tearDown(self):
        """Unset test parameter values."""
        self.data1 = None
        self.data2 = None
        self.rw = None

    def test_cwbreweight(self):
        """Test cwbReweight."""
        npt.assert_array_equal(
            self.rw.weights,
            self.data2,
            err_msg='Incorrect CWB re-weighting.',
        )

        npt.assert_raises(ValueError, self.rw.reweight, self.data1[0])
