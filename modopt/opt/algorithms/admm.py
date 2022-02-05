# -*- coding: utf-8 -*-
"""ADMM algorithms."""

import numpy as np

from modopt.opt.algorithms.base import SetUp


class FastADMM(SetUp):
    r"""Fast ADMM Optimisation.

    This class implement the fast ADMM algorithm, presented in Goldstein 2014.

    Parameters
    ----------
    A : OperatorBase
        Linear operator for x
    B : OperatorBase
        Linear operator for z
    c : array_like
        Constraint vector
    solver1 : function
        Solver for the x update, takes init_value and obs_value as argument.
        ie, return an estimate for:

        .. math:: x_{k+1} = \argmin f(x) + \|A x - y\|
    solver2 : function
        Solver for the z update, takes init_value and obs_value as argument.
        ie return an estimate for:

        .. math:: z_{k+1} = \argmin g(z) + \|Bz - y \|

    rho : float , optional
        regularisation coupling variable default is ``1.0``
    eta : float, optional
        Restart threshold, default is ``0.999``

    Notes
    -----
    The algorithm solve the problem:

    .. math::  x, z = \arg\min f(x) + g(z) + \frac{\rho}{2} \|Ax +Bz - c \|_2^2

    With :math:`f, g` are two convex function (ideally strongly).

    See Also
    --------
    SetUp: parent class
    """

    def __init__(self,
                 x,
                 z,
                 u,
                 A,
                 B,
                 c,
                 solver1,
                 solver2,
                 rho=1,
                 eta=0.9999,
                 max_iter1=5,
                 max_iter2=5,
                 cost=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.A = A
        self.B = B
        self.c = c
        self.solver1 = lambda init, obs: solver1(init, obs, max_iter=max_iter1)
        self.solver2 = lambda init, obs: solver2(init, obs, max_iter=max_iter2)
        self._rho = rho
        self._eta = eta
        self.max_iter1 = max_iter1
        self.max_iter2 = max_iter2

        self._cost_func = cost

        # init iteration variables.
        self._x_old = self.xp.copy(x)
        self._x_new = self.xp.copy(x)
        self._x_hat = self.xp.copy(x)
        self._z_old = self.xp.copy(z)
        self._z_new = self.xp.copy(z)
        self._z_hat = self.xp.copy(z)
        self._u_new = self.xp.copy(u)
        self._u_old = self.xp.copy(u)
        self._u_hat = self.xp.copy(u)

        self._d_old = np.inf
        self._d_new = 0.0
        self._alpha_old = 1
        self._alpha_new = 1

    def _update(self):
        self._x_new = self.solver1(init=self._x_old,
                                   obs=self.B.op(self._z_hat) +
                                   self._u_hat - self.c,
                                   )
        self._u_new = self.A.op(self._x_new)

        self._z_new = self.solver2(init=self._z_hat,
                                   obs=self._u_new +
                                   self._u_hat - self.c,
                                   )

        self._u_new += self._u_old

        self._d_new = np.linalg.norm(self._u_new - self._u_hat) + \
            self._rho * np.linalg.norm(self.B.op(self._z_new - self._z_hat))

        if self._d_new < self._eta * self._d_old:
            print("restart convergence")
            self._alpha_new = (1 + np.sqrt(1 + 4 * self._alpha_old ** 2)) / 2
            update = (self._alpha_old - 1) / self._alpha_new
            self._z_hat = self._z_new + update * (self._z_new - self._z_old)
            self._u_hat = self._u_new + update * (self._u_new - self._u_old)
            self._d_old = self._d_new

        else:
            # restart
            self._alpha_new = 1
            self.xp.copyto(self._z_hat, self._z_new)
            self.xp.copyto(self._u_hat, self._u_new)
            self._d_new = self._d_old / self._eta

        self._alpha_old = self._alpha_new
        self.xp.copyto(self._x_old, self._x_new)
        self.xp.copyto(self._z_old, self._z_new)
        self.xp.copyto(self._u_old, self._u_new)

        # Test cost function for convergence.
        if self._cost_func:
            self.converge = (
                self.any_convergence_flag()
                or self._cost_func.get_cost(self._x_new)
            )

    def iterate(self, max_iter=15):
        """Iterate.

        This method calls update until either convergence criteria is met or
        the maximum number of iterations is reached.

        Parameters
        ----------
        max_iter : int, optional
            Maximum number of iterations (default is ``150``)

        """
        self._run_alg(max_iter)

        # retrieve metrics results
        self.retrieve_outputs()
        # rename outputs as attributes
        self.x_final = self._x_new
        self.z_final = self._z_new

    def get_notify_observers_kwargs(self):
        """Notify observers.

        Return the mapping between the metrics call and the iterated
        variables.

        Returns
        -------
        dict
           The mapping between the iterated variables

        """
        return {
            'x_new': self._x_new,
            'z_new': self._z,
            'idx': self.idx,
        }

    def retrieve_outputs(self):
        """Retrieve outputs.

        Declare the outputs of the algorithms as attributes: x_final,
        y_final, metrics.

        """
        metrics = {}
        for obs in self._observers['cv_metrics']:
            metrics[obs.name] = obs.retrieve_metrics()
        self.metrics = metrics
