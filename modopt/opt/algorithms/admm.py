"""ADMM Algorithms."""
import numpy as np

from modopt.base.backend import get_array_module
from modopt.opt.algorithms.base import SetUp
from modopt.opt.cost import AbstractcostObj


class ADMMcostObj(AbstractcostObj):
    r"""Cost Object for the ADMM problem class.

    Compute :math:`f(u)+g(v) + \tau \| Au +Bv - b\|^2`

    Parameters
    ----------
    cost_funcs: 2-tuples of callable
        f and g function.
    A : OperatorBase
        First Operator
    B : OperatorBase
        Second Operator
    b : array_like
        Observed data
    **kwargs : dict
        Extra parameters for cost operator configuration


    See Also
    --------
    AbstractcostObj: parent class
    """

    def __init__(self, cost_funcs, A, B, b, tau, **kwargs):
        super().__init__(*kwargs)
        self.cost_funcs = cost_funcs
        self.A = A
        self.B = B
        self.b = b
        self.tau = tau

    def _calc_cost(self, u, v, **kwargs):
        """Calculate the cost.

        This method calculates the cost from each of the input operators.

        Parameters
        ----------
        u: array_like
            First primal variable of ADMM
        v: array_like
            Second primal variable of ADMM

        Returns
        -------
        float
            Cost value

        """
        xp = get_array_module(u)
        cost = self.cost_funcs[0](u)
        cost += self.cost_funcs[1](v)
        cost += self.tau * xp.linalg.norm(self.A.op(u) + self.B.op(v) - self.b)
        return cost


class ADMM(SetUp):
    r"""Fast ADMM Optimisation Algorihm.

    This class implement the ADMM algorithm described in :cite:`Goldstein2014` (Algorithm 1).

    Parameters
    ----------
    u: array_like
        First primal variable of ADMM
    v: array_like
        Second primal variable of ADMM
    mu: array_like
        Lagrangian multiplier.
    A : OperatorBase
        Linear operator for u
    B : OperatorBase
        Linear operator for v
    b : array_like
        Constraint vector
    optimizers: 2-tuple of functions
        Solvers for the u and v update, takes init_value and obs_value as
        argument. each element returns an estimate for:
        .. math:: u_{k+1} = \argmin H(u) + \frac{\tau}{2}\|A u - y\|^2
        .. math:: v_{k+1} = \argmin G(v) + \frac{\tau}{2}\|Bv - y \|^2
    cost_funcs = 2-tuple of function
        Compute the values of H and G
    tau: float, default=1
        Coupling parameter for ADMM.
    max_iter2: int

    Notes
    -----
    The algorithm solve the problem:

    .. math:: u, v = \arg\min H(u) + G(v) + \frac\tau2 \|Au + Bv - b \|_2^2

    with the following augmented lagrangian:

    .. math :: \mathcal{L}_{\tau}(u,v, \lambda) = H(u) + G(v)
            +\langle\lambda |Au + Bv -b \rangle + \frac\tau2 \| Au + Bv -b \|^2

    To allow easy iterative solving, the change of variable
    :math:`\mu=\lambda/\tau` is used. Hence, the lagrangian of interest is:

    .. math :: \tilde{\mathcal{L}}_{\tau}(u,v, \mu) = H(u) + G(v)
            + \frac\tau2 \left(\|\mu + Au +Bv - b\|^2 - \|\mu\|^2\right)

    See Also
    --------
    SetUp: parent class
    """

    def __init__(
        self,
        u,
        v,
        mu,
        A,
        B,
        b,
        optimizers,
        tau=1,
        cost_funcs=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.A = A
        self.B = B
        self.b = b
        self._opti_H = optimizers[0]
        self._opti_G = optimizers[1]
        self._tau = tau
        if cost_funcs is not None:
            self._cost_func = ADMMcostObj(cost_funcs, A, B, b, tau)
        else:
            self._cost_func = None

        # init iteration variables.
        self._u_old = self.xp.copy(u)
        self._u_new = self.xp.copy(u)
        self._v_old = self.xp.copy(v)
        self._v_new = self.xp.copy(v)
        self._mu_new = self.xp.copy(mu)
        self._mu_old = self.xp.copy(mu)

    def _update(self):
        self._u_new = self._opti_H(
            init=self._u_old,
            obs=self.B.op(self._v_old) + self._u_old - self.b,
        )
        tmp = self.A.op(self._u_new)
        self._v_new = self._opti_G(
            init=self._v_old,
            obs=tmp + self._u_old - self.b,
        )

        self._mu_new = self._mu_old + (tmp + self.B.op(self._v_new) - self.b)

        # update cycle
        self._u_old = self.xp.copy(self._u_new)
        self._v_old = self.xp.copy(self._v_new)
        self._mu_old = self.xp.copy(self._mu_new)

        # Test cost function for convergence.
        if self._cost_func:
            self.converge = self.any_convergence_flag()
            self.converge |= self._cost_func.get_cost(self._u_new, self._v_new)

    def iterate(self, max_iter=150):
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
        self.u_final = self._u_new
        self.x_final = self.u_final # for backward compatibility
        self.v_final = self._v_new

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
            'x_new': self._u_new,
            'v_new': self._v_new,
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


class FastADMM(ADMM):
    r"""Fast ADMM Optimisation Algorihm.

    This class implement the fast ADMM algorithm
    (Algorithm 8 from :cite:`Goldstein2014`)

    Parameters
    ----------
    A : OperatorBase
        Linear operator for u
    B : OperatorBase
        Linear operator for v
    b : array_like
        Constraint vector
    solver1 : function
        Solver for the x update, takes init_value and obs_value as argument.
        ie, return an estimate for:
        .. math:: u_{k+1} = \argmin H(u) + \frac{\tau}{2}\|A u - y\|^2
    solver2 : function
        Solver for the z update, takes init_value and obs_value as argument.
        ie return an estimate for:
        .. math:: v_{k+1} = \argmin G(v) + \frac{\tau}{2}\|Bv - y \|^2
    rho : float , optional
        regularisation coupling variable default is ``1.0``
    eta : float, optional
        Restart threshold, default is ``0.999``

    Notes
    -----
    The algorithm solve the problem:

    .. math:: u, v = \arg\min H(u) + G(v) + \frac\tau2 \|Au + Bv - b \|_2^2

    with the following augmented lagrangian:

    .. math:: \mathcal{L}_{\tau}(u,v, \lambda) = H(u) + G(v)
            +\langle\lambda |Au + Bv -b \rangle + \frac\tau2 \| Au + Bv -b \|^2

    To allow easy iterative solving, the change of variable
    :math:`\mu=\lambda/\tau` is used. Hence, the lagrangian of interest is:

    .. math :: \tilde{\mathcal{L}}_{\tau}(u,v, \mu) = H(u) + G(v)
            + \frac\tau2 \left(\|\mu + Au +Bv - b\|^2 - \|\mu\|^2\right)

    See Also
    --------
    SetUp: parent class
    """

    def __init__(
        self,
        u,
        v,
        mu,
        A,
        B,
        b,
        opti_H,
        opti_G,
        alpha=1,
        eta=0.999,
        tau=1,
        opti_H_kwargs=None,
        opti_G_kwargs=None,
        cost=None,
        **kwargs,
    ):
        super().__init__(
            u=u,
            v=b,
            mu=mu,
            A=A,
            B=B,
            b=b,
            opti_H=opti_H,
            opti_G=opti_G,
            opti_H_kwargs=opti_H_kwargs,
            opti_G_kwargs=opti_G,
            cost=None,
            **kwargs,
        )
        self._c_old = np.inf
        self._c_new = 0
        self._eta = eta
        self._alpha_old = alpha
        self._alpha_new = alpha
        self._v_hat = self.xp.copy(self._v_new)
        self._mu_hat = self.xp.copy(self._mu_new)

    def _update(self):
        # Classical ADMM steps
        self._u_new = self._opti_H(
            init=self._u_old,
            obs=self.B.op(self._v_hat) + self._u_old - self.b,
        )
        tmp = self.A.op(self._u_new)
        self._v_new = self._opti_G(
            init=self._v_hat,
            obs=tmp + self._u_hat - self.b,
        )

        self._mu_new = self._mu_hat + (tmp + self.B.op(self._v_new) - self.b)

        # restarting condition
        self._c_new = self.xp.linalg.norm(self._mu_new - self._mu_hat)
        self._c_new += self._tau * self.xp.linalg.norm(
            self.B.op(self._v_new - self._v_hat),
        )
        if self._c_new < self._eta * self._c_old:
            self._alpha_new = 1 + np.sqrt(1 + 4 * self._alpha_old**2)
            beta = (self._alpha_new - 1) / self._alpha_old
            self._v_hat = self._v_new + (self._v_new - self._v_old) * beta
            self._mu_hat = self._mu_new + (self._mu_new - self._mu_old) * beta
        else:
            # reboot to old iteration
            self._alpha_new = 1
            self._v_hat = self._v_old
            self._mu_hat = self._mu_old
            self._c_new = self._c_old / self._eta

        self.xp.copyto(self._u_old, self._u_new)
        self.xp.copyto(self._v_old, self._v_new)
        self.xp.copyto(self._mu_old, self._mu_new)
        # Test cost function for convergence.
        if self._cost_func:
            self.converge = self.any_convergence_flag()
            self.convergd |= self._cost_func.get_cost(self._u_new, self._v_new)
