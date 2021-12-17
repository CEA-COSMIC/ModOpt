# -*- coding: utf-8 -*-
r"""OPTIMISATION ALGOTITHMS.

This module contains class implementations of various optimisation algoritms.

:Authors:

* Samuel Farrens <samuel.farrens@cea.fr>,
* Zaccharie Ramzi <zaccharie.ramzi@cea.fr>,
* Pierre-Antoine Comby <pierre-antoine.comby@ens-paris-saclay.fr>

:Notes:

Input classes must have the following properties:

* **Gradient Operators**

Must have the following methods:

    * ``get_grad()`` - calculate the gradient

Must have the following variables:

    * ``grad`` - the gradient

* **Linear Operators**

Must have the following methods:

    * ``op()`` - operator
    * ``adj_op()`` - adjoint operator

* **Proximity Operators**

Must have the following methods:

    * ``op()`` - operator

The following notation is used to implement the algorithms:

    * ``x_old`` is used in place of :math:`x_{n}`.
    * ``x_new`` is used in place of :math:`x_{n+1}`.
    * ``x_prox`` is used in place of :math:`\tilde{x}_{n+1}`.
    * ``x_temp`` is used for intermediate operations.

"""

from modopt.opt.algorithms.base import SetUp
from modopt.opt.algorithms.forward_backward import (FISTA, POGM,
                                                    ForwardBackward,
                                                    GenForwardBackward)
from modopt.opt.algorithms.gradient_descent import (AdaGenericGradOpt,
                                                    ADAMGradOpt,
                                                    GenericGradOpt,
                                                    MomentumGradOpt,
                                                    RMSpropGradOpt,
                                                    SAGAOptGradOpt,
                                                    VanillaGenericGradOpt)
from modopt.opt.algorithms.primal_dual import Condat
