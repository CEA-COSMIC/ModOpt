"""UNIT TESTS FOR OPT.

This module contains  tests for the modopt.opt module.

:Authors:
    Samuel Farrens <samuel.farrens@cea.fr>
    Pierre-Antoine Comby <pierre-antoine.comby@cea.fr>
"""

import numpy as np
import numpy.testing as npt
import pytest
from pytest_cases import parametrize, parametrize_with_cases, case, fixture, fixture_ref

from modopt.opt import cost, gradient, linear, proximity, reweight

from test_helpers import Dummy

SKLEARN_AVAILABLE = True
try:
    import sklearn
except ImportError:
    SKLEARN_AVAILABLE = False

PTWT_AVAILABLE = True
try:
    import ptwt
    import cupy
except ImportError:
    PTWT_AVAILABLE = False

PYWT_AVAILABLE = True
try:
    import pywt
    import joblib
except ImportError:
    PYWT_AVAILABLE = False

rng = np.random.default_rng()


# Basic functions to be used as operators or as dummy functions
def func_identity(x_val, *args, **kwargs):
    """Return x."""
    return x_val


def func_double(x_val, *args, **kwargs):
    """Double x."""
    return x_val * 2


def func_sq(x_val, *args, **kwargs):
    """Square x."""
    return x_val**2


def func_cube(x_val, *args, **kwargs):
    """Cube x."""
    return x_val**3


@case(tags="cost")
@parametrize(
    ("cost_interval", "n_calls", "converged"),
    [(1, 1, False), (1, 2, True), (2, 5, False), (None, 6, False)],
)
def case_cost_op(cost_interval, n_calls, converged):
    """Case function for costs."""
    dummy_inst1 = Dummy()
    dummy_inst1.cost = func_sq
    dummy_inst2 = Dummy()
    dummy_inst2.cost = func_cube

    cost_obj = cost.costObj([dummy_inst1, dummy_inst2], cost_interval=cost_interval)

    for _ in range(n_calls + 1):
        cost_obj.get_cost(2)
    return cost_obj, converged


@parametrize_with_cases("cost_obj, converged", cases=".", has_tag="cost")
def test_costs(cost_obj, converged):
    """Test cost."""
    npt.assert_equal(cost_obj.get_cost(2), converged)
    if cost_obj._cost_interval:
        npt.assert_equal(cost_obj.cost, 12)


def test_raise_cost():
    """Test error raising for cost."""
    npt.assert_raises(TypeError, cost.costObj, 1)
    npt.assert_raises(ValueError, cost.costObj, [Dummy(), Dummy()])


@case(tags="grad")
@parametrize(call=("op", "trans_op", "trans_op_op"))
def case_grad_parent(call):
    """Case for gradient parent."""
    input_data = np.arange(9).reshape(3, 3)
    callables = {
        "op": func_sq,
        "trans_op": func_cube,
        "get_grad": func_identity,
        "cost": lambda input_val: 1.0,
    }

    grad_op = gradient.GradParent(
        input_data,
        **callables,
        data_type=np.floating,
    )
    if call != "trans_op_op":
        result = callables[call](input_data)
    else:
        result = callables["trans_op"](callables["op"](input_data))

    grad_call = getattr(grad_op, call)(input_data)
    return grad_call, result


@parametrize_with_cases("grad_values, result", cases=".", has_tag="grad")
def test_grad_op(grad_values, result):
    """Test Gradient operator."""
    npt.assert_equal(grad_values, result)


@pytest.fixture
def grad_basic():
    """Case for GradBasic."""
    input_data = np.arange(9).reshape(3, 3)
    grad_op = gradient.GradBasic(
        input_data,
        func_sq,
        func_cube,
        verbose=True,
    )
    grad_op.get_grad(input_data)
    return grad_op


def test_grad_basic(grad_basic):
    """Test grad basic."""
    npt.assert_array_equal(
        grad_basic.grad,
        np.array(
            [
                [0, 0, 8.0],
                [2.16000000e2, 1.72800000e3, 8.0e3],
                [2.70000000e4, 7.40880000e4, 1.75616000e5],
            ]
        ),
        err_msg="Incorrect gradient.",
    )


def test_grad_basic_cost(grad_basic):
    """Test grad_basic cost."""
    npt.assert_almost_equal(grad_basic.cost(np.arange(9).reshape(3, 3)), 3192.0)


def test_grad_op_raises():
    """Test raise error."""
    npt.assert_raises(
        TypeError,
        gradient.GradParent,
        1,
        func_sq,
        func_cube,
    )


#############
# LINEAR OP #
#############


class LinearCases:
    """Linear operator cases."""

    def case_linear_identity(self):
        """Case linear operator identity."""
        linop = linear.Identity()

        data_op, data_adj_op, res_op, res_adj_op = 1, 1, 1, 1

        return linop, data_op, data_adj_op, res_op, res_adj_op

    def case_linear_wavelet_convolve(self):
        """Case linear operator wavelet."""
        linop = linear.WaveletConvolve(
            filters=np.arange(8).reshape(2, 2, 2).astype(float)
        )
        data_op = np.arange(4).reshape(1, 2, 2).astype(float)
        data_adj_op = np.arange(8).reshape(1, 2, 2, 2).astype(float)
        res_op = np.array([[[[0, 0], [0, 4.0]], [[0, 4.0], [8.0, 28.0]]]])
        res_adj_op = np.array([[[28.0, 62.0], [68.0, 140.0]]])

        return linop, data_op, data_adj_op, res_op, res_adj_op

    @parametrize(
        compute_backend=[
            pytest.param(
                "numpy",
                marks=pytest.mark.skipif(
                    not PYWT_AVAILABLE, reason="PyWavelet not available."
                ),
            ),
            pytest.param(
                "cupy",
                marks=pytest.mark.skipif(
                    not PTWT_AVAILABLE, reason="Pytorch Wavelet not available."
                ),
            ),
        ]
    )
    def case_linear_wavelet_transform(self, compute_backend):
        """Case linear wavelet operator."""
        linop = linear.WaveletTransform(
            wavelet_name="haar",
            shape=(8, 8),
            level=2,
        )
        data_op = np.arange(64).reshape(8, 8).astype(float)
        res_op, slices, shapes = pywt.ravel_coeffs(
            pywt.wavedecn(data_op, "haar", level=2)
        )
        data_adj_op = linop.op(data_op)
        res_adj_op = pywt.waverecn(
            pywt.unravel_coeffs(data_adj_op, slices, shapes, "wavedecn"), "haar"
        )
        return linop, data_op, data_adj_op, res_op, res_adj_op

    @parametrize(weights=[[1.0, 1.0], None])
    def case_linear_combo(self, weights):
        """Case linear operator combo with weights."""
        parent = linear.LinearParent(
            func_sq,
            func_cube,
        )
        linop = linear.LinearCombo([parent, parent], weights)

        data_op, data_adj_op, res_op, res_adj_op = (
            2,
            np.array([2, 2]),
            np.array([4, 4]),
            8.0 * (2 if weights else 1),
        )

        return linop, data_op, data_adj_op, res_op, res_adj_op

    @parametrize(factor=[1, 1 + 1j])
    def case_linear_matrix(self, factor):
        """Case linear operator from matrix."""
        linop = linear.MatrixOperator(np.eye(5) * factor)
        data_op = np.arange(5)
        data_adj_op = np.arange(5)
        res_op = np.arange(5) * factor
        res_adj_op = np.arange(5) * np.conjugate(factor)

        return linop, data_op, data_adj_op, res_op, res_adj_op


@fixture
@parametrize_with_cases(
    "linop, data_op, data_adj_op, res_op, res_adj_op", cases=LinearCases
)
def lin_adj_op(linop, data_op, data_adj_op, res_op, res_adj_op):
    """Get adj_op relative data."""
    return linop.adj_op, data_adj_op, res_adj_op


@fixture
@parametrize_with_cases(
    "linop, data_op, data_adj_op, res_op, res_adj_op", cases=LinearCases
)
def lin_op(linop, data_op, data_adj_op, res_op, res_adj_op):
    """Get op relative data."""
    return linop.op, data_op, res_op


@parametrize(
    ("action", "data", "result"), [fixture_ref(lin_op), fixture_ref(lin_adj_op)]
)
def test_linear_operator(action, data, result):
    """Test linear operator."""
    npt.assert_almost_equal(action(data), result)


dummy_with_op = Dummy()
dummy_with_op.op = lambda x: x


@pytest.mark.parametrize(
    ("args", "error"),
    [
        ([linear.LinearParent(func_sq, func_cube)], TypeError),
        ([[]], ValueError),
        ([[Dummy()]], ValueError),
        ([[dummy_with_op]], ValueError),
        ([[]], ValueError),
        ([[linear.LinearParent(func_sq, func_cube)] * 2, [1.0]], ValueError),
        ([[linear.LinearParent(func_sq, func_cube)] * 2, ["1", "1"]], TypeError),
    ],
)
def test_linear_combo_errors(args, error):
    """Test linear combo_errors."""
    npt.assert_raises(error, linear.LinearCombo, *args)


#############
# Proximity #
#############


class ProxCases:
    """Class containing all proximal operator cases.

    Each case should return 4 parameters:
    1. The proximal operator
    2. test input data
    3. Expected result data
    4. Expected cost value.
    """

    weights = np.ones(9).reshape(3, 3).astype(float) * 3
    array33 = np.arange(9).reshape(3, 3).astype(float)
    array33_st = np.array([[-0, -0, -0], [0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    array33_st2 = array33_st * -1

    array33_support = np.asarray([[0, 0, 0], [0, 1.0, 1.25], [1.5, 1.75, 2.0]])

    array233 = np.arange(18).reshape(2, 3, 3).astype(float)
    array233_2 = np.array(
        [
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
        ]
    )
    array233_3 = np.array(
        [
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [
                [4.00795282, 4.60438026, 5.2008077],
                [5.79723515, 6.39366259, 6.99009003],
                [7.58651747, 8.18294492, 8.77937236],
            ],
        ]
    )

    def case_prox_parent(self):
        """Case prox parent."""
        return (
            proximity.ProximityParent(
                func_sq,
                func_double,
            ),
            3,
            9,
            6,
        )

    def case_prox_identity(self):
        """Case prox identity."""
        return proximity.IdentityProx(), 3, 3, 0

    def case_prox_positivity(self):
        """Case prox positivity."""
        return proximity.Positivity(), -3, 0, 0

    def case_prox_sparsethresh(self):
        """Case prox sparsethreshosld."""
        return (
            proximity.SparseThreshold(linear.Identity(), weights=self.weights),
            self.array33,
            self.array33_st,
            108,
        )

    @parametrize(
        "lowr_type, initial_rank, operator, result, cost",
        [
            ("standard", None, None, array233_2, 469.3913294246498),
            ("standard", 1, None, array233_2, 469.3913294246498),
            ("ngole", None, func_double, array233_3, 469.3913294246498),
        ],
    )
    def case_prox_lowrank(self, lowr_type, initial_rank, operator, result, cost):
        """Case prox lowrank."""
        return (
            proximity.LowRankMatrix(
                10,
                lowr_type=lowr_type,
                initial_rank=initial_rank,
                operator=operator,
                thresh_type="hard" if lowr_type == "standard" else "soft",
            ),
            self.array233,
            result,
            cost,
        )

    def case_prox_linear_comp(self):
        """Case prox linear comp."""
        return (
            proximity.LinearCompositionProx(
                linear_op=linear.Identity(), prox_op=self.case_prox_sparsethresh()[0]
            ),
            self.array33,
            self.array33_st,
            108,
        )

    def case_prox_ridge(self):
        """Case prox ridge."""
        return (
            proximity.Ridge(linear.Identity(), self.weights),
            self.array33 * (1 + 1j),
            self.array33 * (1 + 1j) / 7,
            1224,
        )

    @parametrize("alpha, beta", [(0, weights), (weights, 0)])
    def case_prox_elasticnet(self, alpha, beta):
        """Case prox elastic net."""
        if np.all(alpha == 0):
            data = self.case_prox_sparsethresh()[1:]
        else:
            data = self.case_prox_ridge()[1:]
        return (proximity.ElasticNet(linear.Identity(), alpha, beta), *data)

    @parametrize(
        "beta, k_value, data, result, cost",
        [
            (0.2, 1, array33.flatten(), array33_st.flatten(), 259.2),
            (3, 5, array33.flatten(), array33_support.flatten(), 684.0),
            (
                6.0,
                9,
                array33.flatten() * (1 + 1j),
                array33.flatten() * (1 + 1j) / 7,
                1224,
            ),
        ],
    )
    def case_prox_Ksupport(self, beta, k_value, data, result, cost):
        """Case prox K-support norm."""
        return (proximity.KSupportNorm(beta=beta, k_value=k_value), data, result, cost)

    @parametrize(use_weights=[True, False])
    def case_prox_grouplasso(self, use_weights):
        """Case GroupLasso proximity."""
        if use_weights:
            weights = np.tile(self.weights, (4, 1, 1))
        else:
            weights = np.tile(np.zeros((3, 3)), (4, 1, 1))

        random_data = 3 * rng.random(weights[0].shape)
        random_data_tile = np.tile(random_data, (weights.shape[0], 1, 1))
        if use_weights:
            gl_result_data = 2 * random_data_tile - 3
            gl_result_data = (
                np.array(gl_result_data * (gl_result_data > 0).astype("int")) / 2
            )
            cost = np.sum(random_data_tile) * 6
        else:
            gl_result_data = random_data_tile
            cost = 0
        return (
            proximity.GroupLASSO(
                weights=weights,
            ),
            random_data_tile,
            gl_result_data,
            cost,
        )

    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="sklearn not available.")
    def case_prox_owl(self):
        """Case prox for Ordered Weighted L1 Norm."""
        return (
            proximity.OrderedWeightedL1Norm(self.weights.flatten()),
            self.array33.flatten(),
            self.array33_st.flatten(),
            108.0,
        )


@parametrize_with_cases("operator, input_data, op_result, cost_result", cases=ProxCases)
def test_prox_op(operator, input_data, op_result, cost_result):
    """Test proximity operator op."""
    npt.assert_almost_equal(operator.op(input_data), op_result)


@parametrize_with_cases("operator, input_data, op_result, cost_result", cases=ProxCases)
def test_prox_cost(operator, input_data, op_result, cost_result):
    """Test proximity operator cost."""
    npt.assert_almost_equal(operator.cost(input_data, verbose=True), cost_result)


@parametrize(
    "arg, error",
    [
        (1, TypeError),
        ([], ValueError),
        ([Dummy()], ValueError),
        ([dummy_with_op], ValueError),
    ],
)
def test_error_prox_combo(arg, error):
    """Test errors for proximity combo."""
    npt.assert_raises(error, proximity.ProximityCombo, arg)


@pytest.mark.skipif(SKLEARN_AVAILABLE, reason="sklearn is installed")
def test_fail_sklearn():
    """Test fail OWL with sklearn."""
    npt.assert_raises(ImportError, proximity.OrderedWeightedL1Norm, 1)


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="sklearn is not installed.")
def test_fail_owl():
    """Test errors for Ordered Weighted L1 Norm."""
    npt.assert_raises(
        ValueError,
        proximity.OrderedWeightedL1Norm,
        np.arange(10),
    )

    npt.assert_raises(
        ValueError,
        proximity.OrderedWeightedL1Norm,
        -np.arange(10),
    )


def test_fail_lowrank():
    """Test fail for lowrank."""
    prox_op = proximity.LowRankMatrix(10, lowr_type="fail")
    npt.assert_raises(ValueError, prox_op.op, 0)


def test_fail_Ksupport_norm():
    """Test fail for K-support norm."""
    npt.assert_raises(ValueError, proximity.KSupportNorm, 0, 0)


def test_reweight():
    """Test for reweight module."""
    data1 = np.arange(9).reshape(3, 3).astype(float) + 1
    data2 = np.array(
        [[0.5, 1.0, 1.5], [2.0, 2.5, 3.0], [3.5, 4.0, 4.5]],
    )

    rw = reweight.cwbReweight(data1)
    rw.reweight(data1)

    npt.assert_array_equal(
        rw.weights,
        data2,
        err_msg="Incorrect CWB re-weighting.",
    )

    npt.assert_raises(ValueError, rw.reweight, data1[0])
