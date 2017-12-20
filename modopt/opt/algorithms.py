# -*- coding: utf-8 -*-

r"""OPTIMISATION ALGOTITHMS

This module contains class implementations of various optimisation algoritms.

:Author: Samuel Farrens <samuel.farrens@cea.fr>

:Version: 1.4

:Date: 15/12/2017

NOTES
-----

Input classes must have the following properties:

    * **Gradient Operators**

    Must have the following methods:

        * ``get_grad()`` - calculate the gradient

    Must have the following variables:

        * ``grad`` - the gradient
        * ``inv_spec_rad`` - inverse spectral radius :math:`\frac{1}{\rho}`

    * **Linear Operators**

    Must have the following methods:

        * ``op()`` - operator
        * ``adj_op()`` - adjoint operator

    Must have the following variables:

        * ``l1norm`` - the l1 norm of the operator

    * **Proximity Operators**

    Must have the following methods:

        * ``op()`` - operator

The following notation is used to implement the algorithms:

    * x_old is used in place of :math:`x_{n}`.
    * x_new is used in place of :math:`x_{n+1}`.
    * x_prox is used in place of :math:`\tilde{x}_{n+1}`.
    * x_temp is used for intermediate operations.

"""

from __future__ import division, print_function
from builtins import range, zip
import numpy as np
from pydoc import locate
from modopt.interface.errors import warn
from modopt.opt.cost import costObj
from modopt.opt.linear import Identity


class SetUp(object):
    """Algorithm Set-Up

    This class contains methods for checking the set-up of an optimisation
    algotithm and produces warnings if they do not comply

    """

    def _check_sub_class(self, method, parent_class):
        """Check Sub-Class

        This method checks if the input method is an instance or subclass of
        the given parent class

        Parameters
        ----------
        method : str
            Algorithm method to check
        parent_class : class
            Expectec parent class

        """

        if (hasattr(self, method) and not
                issubclass(type(getattr(self, method)), locate(parent_class))):
            warn('{0} provided is not a subclass of {1}.'.format(method,
                 parent_class))

    def _check_set_up(self):
        """ Check Set-Up

        This method checks all possible algorithm methods against the expected
        parent classes

        """

        methods = {'grad': 'modopt.opt.gradient.GradParent',
                   'prox': 'modopt.opt.proximity.ProximityParent',
                   'prox_dual': 'modopt.opt.proximity.ProximityParent',
                   'linear': 'modopt.opt.linear.LinearParent',
                   'cost_func': 'modopt.opt.cost.costObj'}

        for (method, parent_class) in methods.items():
            self._check_sub_class(method, parent_class)


class FISTA(object):
    r"""FISTA

    This class is inhereited by optimisation classes to speed up convergence

    Parameters
    ----------
    lambda_init : float, optional
        Initial value of the relaxation parameter
    active : bool, optional
        Option to activate FISTA convergence speed-up (default is ``True``)

    """

    def __init__(self, lambda_init=None, active=True):

        self.lambda_now = lambda_init
        self.t_now = 1.0
        self.t_prev = 1.0
        self.use_speed_up = active

    def speed_switch(self, turn_on=True):
        r"""Speed swicth

        This method turns on or off the speed-up

        Parameters
        ----------
        turn_on : bool
            Option to turn on speed-up (default is ``True``)

        """

        self.use_speed_up = turn_on

    def update_lambda(self):
        r"""Update lambda

        This method updates the value of lambda

        Notes
        -----
        Implements steps 3 and 4 from algoritm 10.7 in [B2011]_

        """

        self.t_prev = self.t_now
        self.t_now = (1 + np.sqrt(4 * self.t_prev ** 2 + 1)) * 0.5
        self.lambda_now = 1 + (self.t_prev - 1) / self.t_now

    def speed_up(self):
        r"""speed-up

        This method returns the update if the speed-up is active

        """

        if self.use_speed_up:
            self.update_lambda()


class ForwardBackward(FISTA):
    r"""Forward-Backward optimisation

    This class implements standard forward-backward optimisation with an the
    option to use the FISTA speed-up

    Parameters
    ----------
    x : np.ndarray
        Initial guess for the primal variable
    grad : class
        Gradient operator class
    prox : class
        Proximity operator class
    cost : class, optional
        Cost function class
    lambda_init : float, optional
        Initial value of the relaxation parameter
    lambda_update : function, optional
        Relaxation parameter update method
    use_fista : bool, optional
        Option to use FISTA (default is ``True``)
    auto_iterate : bool, optional
        Option to automatically begin iterations upon initialisation (default
        is 'True')

    """

    def __init__(self, x, grad, prox, cost=None, lambda_init=None,
                 lambda_update=None, use_fista=True, auto_iterate=True):

        FISTA.__init__(self, lambda_init, use_fista)
        self.x_old = x
        self.z_old = np.copy(self.x_old)
        self.grad = grad
        self.prox = prox
        self.cost_func = cost
        self.lambda_update = lambda_update
        self.converge = False
        if auto_iterate:
            self.iterate()

    def update(self):
        r"""Update

        This method updates the current reconstruction

        Notes
        -----
        Implements algorithm 10.7 (or 10.5) from [B2011]_

        """

        # Step 1 from alg.10.7.
        self.grad.get_grad(self.z_old)
        y_old = self.z_old - self.grad.inv_spec_rad * self.grad.grad

        # Step 2 from alg.10.7.
        self.x_new = self.prox.op(y_old)

        # Steps 3 and 4 from alg.10.7.
        self.speed_up()

        # Step 5 from alg.10.7.
        self.z_new = self.x_old + self.lambda_now * (self.x_new - self.x_old)

        # Test primal variable for convergence.
        if np.sum(np.abs(self.z_old - self.z_new)) <= 1e-6:
            print(' - converged!')
            self.converge = True

        # Update old values for next iteration.
        np.copyto(self.x_old, self.x_new)
        np.copyto(self.z_old, self.z_new)

        # Update parameter values for next iteration.
        if not isinstance(self.lambda_update, type(None)):
            self.lambda_now = self.lambda_update(self.lambda_now)

        # Test cost function for convergence.
        if not isinstance(self.cost_func, type(None)):
            self.converge = self.cost_func.get_cost(self.z_new)

    def iterate(self, max_iter=150):
        r"""Iterate

        This method calls update until either convergence criteria is met or
        the maximum number of iterations is reached

        Parameters
        ----------
        max_iter : int, optional
            Maximum number of iterations (default is ``150``)

        """

        for i in range(max_iter):
            self.update()

            if self.converge:
                print(' - Converged!')
                break

        self.x_final = self.z_new


class GenForwardBackward(object):
    r"""Generalized Forward-Backward optimisation

    This class implements algorithm 1 from [R2012]_

    Parameters
    ----------
    x : np.ndarray
        Initial guess for the primal variable
    grad : class
        Gradient operator class
    prox_list : list
        List of proximity operator classes
    cost : class, optional
        Cost function class
    lambda_init : float, optional
        Initial value of the relaxation parameter
    lambda_update : function, optional
        Relaxation parameter update method
    weights : np.ndarray, optional
        Proximity operator weights
    auto_iterate : bool, optional
        Option to automatically begin iterations upon initialisation (default
        is 'True')

    """

    def __init__(self, x, grad, prox_list, cost=None, lambda_init=1.0,
                 lambda_update=None, weights=None, auto_iterate=True):

        self.x_old = x
        self.grad = grad
        self.prox_list = np.array(prox_list)
        self.cost_func = cost
        self.lambda_init = lambda_init
        self.lambda_update = lambda_update

        if isinstance(weights, type(None)):
            self.weights = np.repeat(1.0 / self.prox_list.size,
                                     self.prox_list.size)
        else:
            self.weights = np.array(weights)

        # Check weights.
        if np.sum(self.weights) != 1.0:
            raise ValueError('Proximity operator weights must sum to 1.0.'
                             'Current sum of weights = ' +
                             str(np.sum(self.weights)))

        self.z = np.array([self.x_old for i in range(self.prox_list.size)])

        self.converge = False
        if auto_iterate:
            self.iterate()

    def update(self):
        r"""Update

        This method updates the current reconstruction

        Notes
        -----
        Implements algorithm 1 from [R2012]_

        """

        # Calculate gradient for current iteration.
        self.grad.get_grad(self.x_old)

        # Update z values.
        for i in range(self.prox_list.size):
            z_temp = (2 * self.x_old - self.z[i] - self.grad.inv_spec_rad *
                      self.grad.grad)
            z_prox = self.prox_list[i].op(z_temp,
                                          extra_factor=self.grad.inv_spec_rad /
                                          self.weights[i])
            self.z[i] += self.lambda_init * (z_prox - self.x_old)

        # Update current reconstruction.
        self.x_new = np.sum((z_i * w_i for z_i, w_i in
                            zip(self.z, self.weights)), axis=0)

        # Update old values for next iteration.
        np.copyto(self.x_old, self.x_new)

        # Update parameter values for next iteration.
        if not isinstance(self.lambda_update, type(None)):
            self.lambda_now = self.lambda_update(self.lambda_now)

        # Test cost function for convergence.
        if not isinstance(self.cost_func, type(None)):
            self.converge = self.cost_func.get_cost(self.x_new)

    def iterate(self, max_iter=150):
        r"""Iterate

        This method calls update until either convergence criteria is met or
        the maximum number of iterations is reached

        Parameters
        ----------
        max_iter : int, optional
            Maximum number of iterations (default is ``150``)

        """

        for i in range(max_iter):
            self.update()

            if self.converge:
                print(' - Converged!')
                break

        self.x_final = self.x_new


class Condat(SetUp):
    r"""Condat optimisation

    This class implements algorithm 10.7 from [Con2013]_

    Parameters
    ----------
    x : np.ndarray
        Initial guess for the primal variable
    y : np.ndarray
        Initial guess for the dual variable
    grad : class
        Gradient operator class
    prox : class
        Proximity primal operator class
    prox_dual : class
        Proximity dual operator class
    linear : class, optional
        Linear operator class (default is None)
    cost : class, optional
        Cost function class (default is None)
    rho : float, optional
        Relaxation parameter (default is 0.5)
    sigma : float, optional
        Proximal dual parameter (default is 1.0)
    tau : float, optional
        Proximal primal paramater (default is 1.0)
    rho_update : function, optional
        Relaxation parameter update method (default is None)
    sigma_update : function, optional
        Proximal dual parameter update method (default is None)
    tau_update : function, optional
        Proximal primal parameter update method (default is None)
    auto_iterate : bool, optional
        Option to automatically begin iterations upon initialisation (default
        is 'True')

    """

    def __init__(self, x, y, grad, prox, prox_dual, linear=None, cost=None,
                 rho=0.5,  sigma=1.0, tau=1.0, rho_update=None,
                 sigma_update=None, tau_update=None, auto_iterate=True):

        self.x_old = np.copy(x)
        self.y_old = np.copy(y)
        self.grad = grad
        self.prox = prox
        self.prox_dual = prox_dual
        if isinstance(linear, type(None)):
            self.linear = Identity()
        else:
            self.linear = linear
        if isinstance(cost, type(None)):
            self.cost_func = costObj([self.grad, self.prox, self.prox_dual])
        else:
            self.cost_func = cost
        self.rho = rho
        self.sigma = sigma
        self.tau = tau
        self.rho_update = rho_update
        self.sigma_update = sigma_update
        self.tau_update = tau_update
        self.converge = False
        self._check_set_up()
        if auto_iterate:
            self.iterate()

    def _update_param(self):
        r"""Update parameters

        This method updates the values of ``rho``, ``sigma`` and ``tau`` with
        the methods provided

        """

        # Update relaxation parameter.
        if not isinstance(self.rho_update, type(None)):
            self.rho = self.rho_update(self.rho)

        # Update proximal dual parameter.
        if not isinstance(self.sigma_update, type(None)):
            self.sigma = self.sigma_update(self.sigma)

        # Update proximal primal parameter.
        if not isinstance(self.tau_update, type(None)):
            self.tau = self.tau_update(self.tau)

    def _update(self):
        r"""Update

        This method updates the current reconstruction

        Notes
        -----
        Implements equation 9 (algorithm 3.1) from [Con2013]_

        - primal proximity operator set up for positivity constraint

        """

        # Step 1 from eq.9.
        self.grad.get_grad(self.x_old)

        x_prox = self.prox.op(self.x_old - self.tau * self.grad.grad -
                              self.tau * self.linear.adj_op(self.y_old))

        # Step 2 from eq.9.
        y_temp = (self.y_old + self.sigma *
                  self.linear.op(2 * x_prox - self.x_old))

        y_prox = (y_temp - self.sigma * self.prox_dual.op(y_temp / self.sigma,
                  extra_factor=(1.0 / self.sigma)))

        # Step 3 from eq.9.
        self.x_new = self.rho * x_prox + (1 - self.rho) * self.x_old
        self.y_new = self.rho * y_prox + (1 - self.rho) * self.y_old

        del x_prox, y_prox, y_temp

        # Update old values for next iteration.
        np.copyto(self.x_old, self.x_new)
        np.copyto(self.y_old, self.y_new)

        # Update parameter values for next iteration.
        self._update_param()

        # Test cost function for convergence.
        self.converge = self.cost_func.get_cost(self.x_new, self.y_new)

    def iterate(self, max_iter=150):
        r"""Iterate

        This method calls update until either convergence criteria is met or
        the maximum number of iterations is reached

        Parameters
        ----------
        max_iter : int, optional
            Maximum number of iterations (default is ``150``)

        """

        for i in range(max_iter):
            self._update()

            if self.converge:
                print(' - Converged!')
                break

        self.x_final = self.x_new
        self.y_final = self.y_new
