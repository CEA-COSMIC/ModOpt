"""UNIT TESTS FOR Algorithms.

This module contains unit tests for the modopt.opt module.

:Authors:
    Samuel Farrens <samuel.farrens@cea.fr>
    Pierre-Antoine Comby <pierre-antoine.comby@cea.fr>
"""

import numpy as np
import numpy.testing as npt
from modopt.opt import algorithms, cost, gradient, linear, proximity, reweight
from pytest_cases import (
    fixture,
    parametrize,
    parametrize_with_cases,
)


SKLEARN_AVAILABLE = True
try:
    import sklearn
except ImportError:
    SKLEARN_AVAILABLE = False


rng = np.random.default_rng()


@fixture
def idty():
    """Identity function."""
    return lambda x: x


@fixture
def reweight_op():
    """Reweight operator."""
    data3 = np.arange(9).reshape(3, 3).astype(float) + 1
    return reweight.cwbReweight(data3)


def build_kwargs(kwargs, use_metrics):
    """Build the kwargs for each algorithm, replacing placeholders by true values.

    This function has to be call for each test, as direct parameterization somehow
    is not working with pytest-xdist and pytest-cases.
    It also adds dummy metric measurement to validate the metric api.
    """
    update_value = {
        "idty": lambda x: x,
        "lin_idty": linear.Identity(),
        "reweight_op": reweight.cwbReweight(
            np.arange(9).reshape(3, 3).astype(float) + 1
        ),
    }
    new_kwargs = dict()
    print(kwargs)
    # update the value of the dict is possible.
    for key in kwargs:
        new_kwargs[key] = update_value.get(kwargs[key], kwargs[key])

    if use_metrics:
        new_kwargs["linear"] = linear.Identity()
        new_kwargs["metrics"] = {
            "diff": {
                "metric": lambda test, ref: np.sum(test - ref),
                "mapping": {"x_new": "test"},
                "cst_kwargs": {"ref": np.arange(9).reshape((3, 3))},
                "early_stopping": False,
            }
        }

    return new_kwargs


@parametrize(use_metrics=[True, False])
class AlgoCases:
    r"""Cases for algorithms.

    Most of the test solves the trivial problem

    .. math::
        \\min_x \\frac{1}{2} \\| y - x \\|_2^2 \\quad\\text{s.t.} x \\geq 0

    More complex and concrete usecases are shown in examples.
    """

    data1 = np.arange(9).reshape(3, 3).astype(float)
    data2 = data1 + rng.standard_normal(data1.shape) * 1e-6
    max_iter = 20

    @parametrize(
        kwargs=[
            {"beta_update": "idty", "auto_iterate": False, "cost": None},
            {"beta_update": "idty"},
            {"cost": None, "lambda_update": None},
            {"beta_update": "idty", "a_cd": 3},
            {"beta_update": "idty", "r_lazy": 3, "p_lazy": 0.7, "q_lazy": 0.7},
            {"restart_strategy": "adaptive", "xi_restart": 0.9},
            {
                "restart_strategy": "greedy",
                "xi_restart": 0.9,
                "min_beta": 1.0,
                "s_greedy": 1.1,
            },
        ]
    )
    def case_forward_backward(self, kwargs, idty, use_metrics):
        """Forward Backward case."""
        update_kwargs = build_kwargs(kwargs, use_metrics)
        algo = algorithms.ForwardBackward(
            self.data1,
            grad=gradient.GradBasic(self.data1, idty, idty),
            prox=proximity.Positivity(),
            **update_kwargs,
        )
        if update_kwargs.get("auto_iterate", None) is False:
            algo.iterate(self.max_iter)
        return algo, update_kwargs

    @parametrize(
        kwargs=[
            {
                "cost": None,
                "auto_iterate": False,
                "gamma_update": "idty",
                "beta_update": "idty",
            },
            {"gamma_update": "idty", "lambda_update": "idty"},
            {"cost": True},
            {"cost": True, "step_size": 2},
        ]
    )
    def case_gen_forward_backward(self, kwargs, use_metrics, idty):
        """General FB setup."""
        update_kwargs = build_kwargs(kwargs, use_metrics)
        grad_inst = gradient.GradBasic(self.data1, idty, idty)
        prox_inst = proximity.Positivity()
        prox_dual_inst = proximity.IdentityProx()
        if update_kwargs.get("cost", None) is True:
            update_kwargs["cost"] = cost.costObj([grad_inst, prox_inst, prox_dual_inst])
        algo = algorithms.GenForwardBackward(
            self.data1,
            grad=grad_inst,
            prox_list=[prox_inst, prox_dual_inst],
            **update_kwargs,
        )
        if update_kwargs.get("auto_iterate", None) is False:
            algo.iterate(self.max_iter)
        return algo, update_kwargs

    @parametrize(
        kwargs=[
            {
                "sigma_dual": "idty",
                "tau_update": "idty",
                "rho_update": "idty",
                "auto_iterate": False,
            },
            {
                "sigma_dual": "idty",
                "tau_update": "idty",
                "rho_update": "idty",
            },
            {
                "linear": "lin_idty",
                "cost": True,
                "reweight": "reweight_op",
            },
        ]
    )
    def case_condat(self, kwargs, use_metrics, idty):
        """Condat Vu Algorithm setup."""
        update_kwargs = build_kwargs(kwargs, use_metrics)
        grad_inst = gradient.GradBasic(self.data1, idty, idty)
        prox_inst = proximity.Positivity()
        prox_dual_inst = proximity.IdentityProx()
        if update_kwargs.get("cost", None) is True:
            update_kwargs["cost"] = cost.costObj([grad_inst, prox_inst, prox_dual_inst])

        algo = algorithms.Condat(
            self.data1,
            self.data2,
            grad=grad_inst,
            prox=prox_inst,
            prox_dual=prox_dual_inst,
            **update_kwargs,
        )
        if update_kwargs.get("auto_iterate", None) is False:
            algo.iterate(self.max_iter)
        return algo, update_kwargs

    @parametrize(kwargs=[{"auto_iterate": False, "cost": None}, {}])
    def case_pogm(self, kwargs, use_metrics, idty):
        """POGM setup."""
        update_kwargs = build_kwargs(kwargs, use_metrics)
        grad_inst = gradient.GradBasic(self.data1, idty, idty)
        prox_inst = proximity.Positivity()
        algo = algorithms.POGM(
            u=self.data1,
            x=self.data1,
            y=self.data1,
            z=self.data1,
            grad=grad_inst,
            prox=prox_inst,
            **update_kwargs,
        )

        if update_kwargs.get("auto_iterate", None) is False:
            algo.iterate(self.max_iter)
        return algo, update_kwargs

    @parametrize(
        GradDescent=[
            algorithms.VanillaGenericGradOpt,
            algorithms.AdaGenericGradOpt,
            algorithms.ADAMGradOpt,
            algorithms.MomentumGradOpt,
            algorithms.RMSpropGradOpt,
            algorithms.SAGAOptGradOpt,
        ]
    )
    def case_grad(self, GradDescent, use_metrics, idty):
        """Gradient Descent algorithm test."""
        update_kwargs = build_kwargs({}, use_metrics)
        grad_inst = gradient.GradBasic(self.data1, idty, idty)
        prox_inst = proximity.Positivity()
        cost_inst = cost.costObj([grad_inst, prox_inst])

        algo = GradDescent(
            self.data1,
            grad=grad_inst,
            prox=prox_inst,
            cost=cost_inst,
            **update_kwargs,
        )
        algo.iterate()
        return algo, update_kwargs

    @parametrize(admm=[algorithms.ADMM, algorithms.FastADMM])
    def case_admm(self, admm, use_metrics, idty):
        """ADMM setup."""

        def optim1(init, obs):
            return obs

        def optim2(init, obs):
            return obs

        update_kwargs = build_kwargs({}, use_metrics)
        algo = admm(
            u=self.data1,
            v=self.data1,
            mu=np.zeros_like(self.data1),
            A=linear.Identity(),
            B=linear.Identity(),
            b=self.data1,
            optimizers=(optim1, optim2),
            **update_kwargs,
        )
        algo.iterate()
        return algo, update_kwargs


@parametrize_with_cases("algo, kwargs", cases=AlgoCases)
def test_algo(algo, kwargs):
    """Test algorithms."""
    if kwargs.get("auto_iterate") is False:
        # algo already run
        npt.assert_almost_equal(algo.idx, AlgoCases.max_iter - 1)
    else:
        npt.assert_almost_equal(algo.x_final, AlgoCases.data1)

    if kwargs.get("metrics"):
        print(algo.metrics)
        npt.assert_almost_equal(algo.metrics["diff"]["values"][-1], 0, 3)
