from modopt.opt.algorithms import SetUp, POGM

from modopt.opt.linear import Identity
from modopt.opt.gradient import GradBasic


class FastADMM(SetUp):
    """ Fast ADMM Optimisation

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

    .. math::  x, z = \arg\min f(x) + g(z) + \frac{\rho2} \|Ax +Bz - c \|_2^2

    With :math:`f, g` are two convex function (ideally strongly).

    See Also
    --------
    SetUp: parent class
    """

    def __init__(self, A, B, c, solver1, solver2, rho=1, eta=0.9999, max_iter1=5, max_iter2=5, **kwargs):
          super().__init__(**kwargs)

        self.A = A
        self.B = B
        self.c = c
        self.solver1 = lambda init, obs: solver1(init, obs, max_iter=max_iter1)
        self.solver2 = lambda init, obs: solver2(init, obs, max_iter=max_iter1)
        self.rho = rho

    def _update(self):
        self._x_new = self.solver1(init_value=self._x_old,
                                   obs_value=(self.B.op(self._z_hat)+self._u_hat-self.c ),
                                   max_iter=

        )

        self._z_new = self.solver1(init_value=self._z_hat, obs_value=(
            self.A.op(self._x_new) + self._u_hat - self.c)
        )

        self._u_new = self._u_old + A.op(self._x_new)


        d_new = np.linalg.norm(u_new - u_hat) + rho * np.linalg.norm(B.op(z_new-z_hat))
        if d_new < eta* d_old:
            alpha_new = (1+np.sqrt(1 + 4 * alpha_old**2)) / 2
            z_hat = z_new + ( alpha_old - 1 ) / alpha_new * (z_new - z_old)
            u_hat = u_new + ( alpha_old - 1 ) / alpha_new * (u_new - u_old)


        else:
            # restart
            alpha_new = 1
            z_hat = z_new.copy()
            u_hat = u_new.copy()
            d_new = d_old / eta

        alpha_old = alpha_new
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
