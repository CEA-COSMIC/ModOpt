"""UNIT TESTS FOR OPT.

This module contains  tests for the modopt.opt module.

:Author: Samuel Farrens <samuel.farrens@cea.fr>
:Author: Pierre-Antoine Comby <pierre-antoine.comby@crans.org>
"""

import numpy as np
import numpy.testing as npt
import pytest
from pytest_cases import parametrize, parametrize_with_cases, case

from modopt.opt import cost, gradient, linear, proximity, reweight

from test_helpers import Dummy

SKLEARN_AVAILABLE = True
try:
    import sklearn
except ImportError:
    SKLEARN_AVAILABLE = False


# Basic functions to be used as operators or as dummy functions
func_identity = lambda x_val: x_val
func_double = lambda x_val: x_val * 2
func_sq = lambda x_val: x_val**2
func_cube = lambda x_val: x_val**3


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
    npt.assert_almost_equal(grad_basic.cost(np.arange(9).reshape(3,3)), 3192.0)



def test_grad_op_raises():
    """Test raise error."""
    npt.assert_raises(
        TypeError,
        gradient.GradParent,
        1,
        func_sq,
        func_cube,
    )
