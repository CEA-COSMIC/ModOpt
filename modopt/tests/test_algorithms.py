# -*- coding: utf-8 -*-

"""UNIT TESTS FOR Algorithms.

This module contains unit tests for the modopt.opt module.

:Author: Samuel Farrens <samuel.farrens@cea.fr>
:Author: Pierre-Antoine Comby <pierre-antoine.comby@crans.org>
"""

import numpy as np
import numpy.testing as npt
from modopt.opt import algorithms, cost, gradient, linear, proximity, reweight
from pytest_cases import (case, fixture, fixture_ref, parametrize,
                          parametrize_with_cases)

from test_helpers import Dummy

SKLEARN_AVAILABLE = True
try:
    import sklearn
except ImportError:
    SKLEARN_AVAILABLE = False


idty = lambda x_val: x_val


class AlgoCases:
    """Cases for algorithms."""

    data1 = np.arange(9).reshape(3, 3).astype(float)
    data2 = data1 + np.random.randn(*data1.shape) * 1e-6
    data3 = np.arange(9).reshape(3, 3).astype(float) + 1
    max_iter = 20

    @parametrize(
        kwargs=[
            {"beta_update": idty, "auto_iterate": False, "cost": None},
            {"beta_update": idty},
            {"cost": None, "lambda_update": None},
            {"beta_update": idty, "a_cd": 3},
            {"beta_update": idty, "r_lazy": 3, "p_lazy": 0.7, "q_lazy": 0.7},
            {"restart_strategy": "adaptive", "xi_restart": 0.9},
            {
                "restart_strategy": "greedy",
                "xi_restart": 0.9,
                "min_beta": 1.0,
                "s_greedy": 1.1,
            },
        ]
    )
    def case_forward_backward(self, kwargs):
        """Forward Backward case."""
        algo = algorithms.ForwardBackward(
            grad=gradient.GradBasic(self.data1, idty, idty),
            prox=proximity.Positivity(),
            **kwargs,
        )
        if kwargs.get("auto_iterate", None) is False:
            algo.iterate(self.max_iter)
        return algo

    @parametrize(
        kwargs=[
            {
                "cost": None,
                "auto_iterate": False,
                "gamma_update": idty,
                "beta_update": idty,
            },
            {"gamma_update": idty, "lambda_update": idty},
            {"cost": True},
            {"cost": True, "step_size": 2},
        ]
    )
    def case_gen_forward_backward(self, kwargs):
        """General FB setup."""
        grad_inst = gradient.GradBasic(self.data1, idty, idty)
        prox_inst = proximity.Positivity()
        prox_dual_inst = proximity.IdentityProx()
        if kwargs.get("cost", None) is True:
            kwargs["cost"] = cost.costObj([grad_inst, prox_inst, prox_dual_inst])
        algo = algorithms.GenForwardBackward(
            grad=grad_inst,
            prox_list=[prox_inst, prox_dual_inst],
            **kwargs,
        )
        if kwargs.get("auto_iterate", None) is False:
            algo.iterate(self.max_iter)
        return algo

    @parametrize(
        kwargs=[
            {
                "sigma_dual": idty,
                "tau_update": idty,
                "rho_update": idty,
                "auto_iterate": False,
            },
            {
                "sigma_dual": idty,
                "tau_update": idty,
                "rho_update": idty,
            },
            {
                "linear": linear.Identity(),
                "cost": True,
                "reweight": reweight.cwbReweight(data3),
            },
        ]
    )
    def case_condat(self, kwargs):
        """Condat Vu Algorithm setup."""
        grad_inst = gradient.GradBasic(self.data1, idty, idty)
        prox_inst = proximity.Positivity()
        prox_dual_inst = proximity.IdentityProx()
        if kwargs.get("cost", None) is True:
            kwargs["cost"] = cost.costObj([grad_inst, prox_inst, prox_dual_inst])

        algo = algorithms.Condat(
            self.data1,
            self.data2,
            grad=grad_inst,
            prox=prox_inst,
            prox_dual=prox_dual_inst,
            **kwargs,
        )
        if kwargs.get("auto_iterate", None) is False:
            algo.iterate(self.max_iter)
        return algo

    @parametrize(kwargs=[{"auto_iterate": False, "cost": None}, {}])
    def case_pogm(self, kwargs):
        """POGM setup."""
        grad_inst = gradient.GradBasic(self.data1, idty, idty)
        prox_inst = proximity.Positivity()
        algo = algorithms.POGM(
            u=self.data1,
            x=self.data1,
            y=self.data1,
            z=self.data1,
            grad=grad_inst,
            prox=prox_inst,
            **kwargs,
        )

        if kwargs.get("auto_iterate", None) is False:
            algo.iterate(self.max_iter)

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
    def case_grad(self, GradDescent):
        """Gradient Descent algorithm test."""
        grad_inst = gradient.GradBasic(self.data1, idty, idty)
        prox_inst = proximity.Positivity()
        cost_inst = cost.costObj([grad_inst, prox_inst])

        algo = GradDescent(
            self.data1,
            grad=grad_inst,
            prox=prox_inst,
            cost=cost_inst,
        )
        return algo
