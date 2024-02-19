"""Primal-Dual Algorithms."""

from modopt.opt.algorithms.base import SetUp
from modopt.opt.cost import costObj
from modopt.opt.linear import Identity


class Condat(SetUp):
    r"""Condat optimisation.

    This class implements algorithm 3.1 from :cite:`condat2013`.

    Parameters
    ----------
    x : numpy.ndarray
        Initial guess for the primal variable
    y : numpy.ndarray
        Initial guess for the dual variable
    grad
        Gradient operator class instance
    prox
        Proximity primal operator class instance
    prox_dual
        Proximity dual operator class instance
    linear : class instance, optional
        Linear operator class instance (default is ``None``)
    cost : class instance or str, optional
        Cost function class instance (default is ``'auto'``); Use ``'auto'`` to
        automatically generate a ``costObj`` instance
    reweight : class instance, optional
        Reweighting class instance
    rho : float, optional
        Relaxation parameter, :math:`\rho` (default is ``0.5``)
    sigma : float, optional
        Proximal dual parameter, :math:`\sigma` (default is ``1.0``)
    tau : float, optional
        Proximal primal paramater, :math:`\tau` (default is ``1.0``)
    rho_update : callable, optional
        Relaxation parameter update method (default is ``None``)
    sigma_update : callable, optional
        Proximal dual parameter update method (default is ``None``)
    tau_update : callable, optional
        Proximal primal parameter update method (default is ``None``)
    auto_iterate : bool, optional
        Option to automatically begin iterations upon initialisation (default
        is ``True``)
    max_iter : int, optional
        Maximum number of iterations (default is ``150``)
    n_rewightings : int, optional
        Number of reweightings to perform (default is ``1``)

    Notes
    -----
    The ``tau_param`` can also be set using the keyword `step_size`, which will
    override the value of ``tau_param``.

    The following state variable are available for metrics measurememts at
    each iteration :

    * ``'x_new'`` : new estimate of :math:`x` (primal variable)
    * ``'y_new'`` : new estimate of :math:`y` (dual variable)
    * ``'idx'`` : index of the iteration.

    See Also
    --------
    modopt.opt.algorithms.base.SetUp : parent class
    modopt.opt.cost.costObj : cost object class
    modopt.opt.gradient : gradient operator classes
    modopt.opt.proximity : proximity operator classes
    modopt.opt.linear : linear operator classes
    modopt.opt.reweight : reweighting classes

    """

    def __init__(
        self,
        x,
        y,
        grad,
        prox,
        prox_dual,
        linear=None,
        cost="auto",
        reweight=None,
        rho=0.5,
        sigma=1.0,
        tau=1.0,
        rho_update=None,
        sigma_update=None,
        tau_update=None,
        auto_iterate=True,
        max_iter=150,
        n_rewightings=1,
        metric_call_period=5,
        metrics=None,
        **kwargs,
    ):
        # Set default algorithm properties
        super().__init__(
            metric_call_period=metric_call_period,
            metrics=metrics,
            **kwargs,
        )

        # Set the initial variable values
        for input_data in (x, y):
            self._check_input_data(input_data)

        self._x_old = self.xp.copy(x)
        self._y_old = self.xp.copy(y)

        # Set the algorithm operators
        for operator in (grad, prox, prox_dual, linear, cost):
            self._check_operator(operator)

        self._grad = grad
        self._prox = prox
        self._prox_dual = prox_dual
        self._reweight = reweight
        if isinstance(linear, type(None)):
            self._linear = Identity()
        else:
            self._linear = linear
        if cost == "auto":
            self._cost_func = costObj(
                [
                    self._grad,
                    self._prox,
                    self._prox_dual,
                ]
            )
        else:
            self._cost_func = cost

        # Set the algorithm parameters
        for param_val in (rho, sigma, tau):
            self._check_param(param_val)

        self._rho = rho
        self._sigma = sigma
        self._tau = self.step_size or tau

        # Set the algorithm parameter update methods
        for param_update in (rho_update, sigma_update, tau_update):
            self._check_param_update(param_update)

        self._rho_update = rho_update
        self._sigma_update = sigma_update
        self._tau_update = tau_update

        # Automatically run the algorithm
        if auto_iterate:
            self.iterate(max_iter=max_iter, n_rewightings=n_rewightings)

    def _update_param(self):
        """Update parameters.

        This method updates the values of the algorthm parameters with the
        methods provided.

        """
        # Update relaxation parameter.
        if not isinstance(self._rho_update, type(None)):
            self._rho = self._rho_update(self._rho)

        # Update proximal dual parameter.
        if not isinstance(self._sigma_update, type(None)):
            self._sigma = self._sigma_update(self._sigma)

        # Update proximal primal parameter.
        if not isinstance(self._tau_update, type(None)):
            self._tau = self._tau_update(self._tau)

    def _update(self):
        """Update.

        This method updates the current reconstruction.

        Notes
        -----
        Implements equation 9 (algorithm 3.1) from :cite:`condat2013`.

        - Primal proximity operator set up for positivity constraint.

        """
        # Step 1 from eq.9.
        self._grad.get_grad(self._x_old)

        x_prox = self._prox.op(
            self._x_old
            - self._tau * self._grad.grad
            - self._tau * self._linear.adj_op(self._y_old),
        )

        # Step 2 from eq.9.
        y_temp = self._y_old + self._sigma * self._linear.op(2 * x_prox - self._x_old)

        y_prox = y_temp - self._sigma * self._prox_dual.op(
            y_temp / self._sigma,
            extra_factor=(1.0 / self._sigma),
        )

        # Step 3 from eq.9.
        self._x_new = self._rho * x_prox + (1 - self._rho) * self._x_old
        self._y_new = self._rho * y_prox + (1 - self._rho) * self._y_old

        del x_prox, y_prox, y_temp

        # Update old values for next iteration.
        self.xp.copyto(self._x_old, self._x_new)
        self.xp.copyto(self._y_old, self._y_new)

        # Update parameter values for next iteration.
        self._update_param()

        # Test cost function for convergence.
        if self._cost_func:
            self.converge = self.any_convergence_flag() or self._cost_func.get_cost(
                self._x_new, self._y_new
            )

    def iterate(self, max_iter=150, n_rewightings=1, progbar=None):
        """Iterate.

        This method calls update until either convergence criteria is met or
        the maximum number of iterations is reached.

        Parameters
        ----------
        max_iter : int, optional
            Maximum number of iterations (default is ``150``)
        n_rewightings : int, optional
            Number of reweightings to perform (default is ``1``)
        progbar: tqdm.tqdm
            Progress bar handle (default is ``None``)
        """
        self._run_alg(max_iter, progbar)

        if not isinstance(self._reweight, type(None)):
            for _ in range(n_rewightings):
                self._reweight.reweight(self._linear.op(self._x_new))
                if progbar:
                    progbar.reset(total=max_iter)
                self._run_alg(max_iter, progbar)

        # retrieve metrics results
        self.retrieve_outputs()
        # rename outputs as attributes
        self.x_final = self._x_new
        self.y_final = self._y_new

    def get_notify_observers_kwargs(self):
        """Notify observers.

        Return the mapping between the metrics call and the iterated
        variables.

        Returns
        -------
        notify_observers_kwargs : dict,
           The mapping between the iterated variables

        """
        return {"x_new": self._x_new, "y_new": self._y_new, "idx": self.idx}

    def retrieve_outputs(self):
        """Retrieve outputs.

        Declare the outputs of the algorithms as attributes: ``x_final``,
        ``y_final``, ``metrics``.

        """
        metrics = {}
        for obs in self._observers["cv_metrics"]:
            metrics[obs.name] = obs.retrieve_metrics()
        self.metrics = metrics
