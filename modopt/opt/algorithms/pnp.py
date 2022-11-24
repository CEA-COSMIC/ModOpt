r"""Plug 'n play algorithms.

These algorithms solves

..math :: \mathrm{arg}\min_{x\in \mathbb{R}^d} f(x) + g(x)

"""

from modopt.opt.algorithms import SetUp


class PnpADMM(SetUp):
    r"""Plug 'n Play ADMM.

    Implements (PNP-ADMM) of :cite:`ryu2019` to solve

    ..math :: \arg\min f(x) + g(x)

    Parameters
    ----------
    x: array
        Initial Value
    proxf: Operator
        Data consistency function as a proximal operator.
    proxg: Operator
        Regularisation function (e.g. denoiser).
    alpha: float, default 1
        Data consistency parameter, analoguous to gradient step size.
    sigma: float,  default 1
        Noise level parameter.
    """

    def __init__(
        self,
        x_init,
        proxf,
        proxg,
        alpha=1,
        sigma=1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # init iteration variables.
        self._x_old = self.xp.copy(x_init)
        self._x_new = self.xp.copy(x_init)
        self._y_old = self.xp.copy(x_init)
        self._y_new = self.xp.copy(x_init)
        self._u_old = self.xp.copy(x_init)
        self._u_new = self.xp.copy(x_init)

        # algorithm parameters
        self._proxf = proxf
        self._proxg = proxg
        self._alpha = alpha
        self._sigma = sigma

    def _update(self):
        self._x_new = self._proxf.op(
            self._y_old - self._u_old,
            extra_factor=self._alpha,
        )
        self._y_new = self._proxg.op(self._x_new - self._u_old)
        self._u_new = self._u_old + self._x_new - self._y_new

        # Update iteration
        self.xp.copyto(self._x_old, self._x_new)
        self.xp.copyto(self._y_old, self._y_new)
        self.xp.copyto(self._u_old, self._u_new)


class PnpFBS(SetUp):
    """Plug'n Play Forward Backward Splitting.

    Implements (PNP-FBS) of :cite:`ryu2019`

    Parameters
    ----------
    x: array
        Initial estimation
    gradf: Operator
        Gradient of :math:`f`
    proxg: Operator
        Proximal operator or plug-in replacement for g
    alpha: float
        gradient descent step size
    sigma: float
        Noise level.
    """

    def __init__(
        self,
        x_init,
        gradf,
        proxg,
        alpha=1,
        sigma=1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # init iteration variables.
        self._x_old = self.xp.copy(x_init)
        self._x_new = self.xp.copy(x_init)

        # algorithm parameters
        self._gradf = gradf
        self._proxg = proxg
        self._alpha = alpha
        self._sigma = sigma

    def _update(self):
        self._gradf.get_grad(self._x_old)
        # saves an (possibly expensive) array allocation
        self._x_new = (-self._alpha) * self._gradf.grad
        self._x_new += self._x_old
        self._x_new = self._proxg.op(self._x_new)

        # Update iteration
        self.xp.copyto(self._x_old, self._x_new)

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
