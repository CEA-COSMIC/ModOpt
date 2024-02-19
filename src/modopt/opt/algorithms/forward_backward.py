"""Forward-Backward Algorithms."""

import numpy as np

from modopt.base import backend
from modopt.opt.algorithms.base import SetUp
from modopt.opt.cost import costObj
from modopt.opt.linear import Identity


class FISTA:
    r"""FISTA.

    This class is inherited by optimisation classes to speed up convergence
    The parameters for the modified FISTA are as described in :cite:`liang2018`
    :math:`(p, q, r)`-lazy or in :cite:`chambolle2015` (a_cd).
    The restarting strategies are those described in :cite:`liang2018`,
    algorithms 4-5.

    Parameters
    ----------
    restart_strategy: str or None
        Name of the restarting strategy, if ``None``, there is no restarting
        (default is ``None``)
    min_beta: float or None
        The minimum :math:`\beta` value when using the greedy restarting
        strategy (default is ``None``)
    s_greedy: float or None
        Parameter for the safeguard comparison in the greedy restarting
        strategy, it must be > 1
        (default is ``None``)
    xi_restart: float or None
        Mutlitplicative parameter for the update of beta in the greedy
        restarting strategy and for the update of r_lazy in the adaptive
        restarting strategies, it must be > 1
        (default is ``None``)
    a_cd: float or None
        Parameter for the update of lambda in Chambolle-Dossal mode, if
        ``None`` the mode of the algorithm is the regular FISTA, else the mode
        is Chambolle-Dossal, it must be > 2
    p_lazy: float
        Parameter for the update of lambda in Fista-Mod, it must satisfy
        :math:`p \in ]0, 1]`
    q_lazy: float
        Parameter for the update of lambda in Fista-Mod, it must satisfy
        :math:`q \in ]0, (2-p)^2]`
    r_lazy: float
        Parameter for the update of lambda in Fista-Mod, it must satisfy
        :math:`r \in ]0, 4]`

    """

    _restarting_strategies = (
        "adaptive",  # option 1 in alg 4
        "adaptive-i",
        "adaptive-1",
        "adaptive-ii",  # option 2 in alg 4
        "adaptive-2",
        "greedy",  # alg 5
        None,  # no restarting
    )

    def __init__(
        self,
        restart_strategy=None,
        min_beta=None,
        s_greedy=None,
        xi_restart=None,
        a_cd=None,
        p_lazy=1,
        q_lazy=1,
        r_lazy=4,
        **kwargs,
    ):
        if isinstance(a_cd, type(None)):
            self.mode = "regular"
            self.p_lazy = p_lazy
            self.q_lazy = q_lazy
            self.r_lazy = r_lazy

        elif a_cd > 2:
            self.mode = "CD"
            self.a_cd = a_cd
            self._n = 0

        else:
            raise ValueError(
                "a_cd must either be None (for regular mode) or a number > 2",
            )

        if restart_strategy in self._restarting_strategies:
            self._check_restart_params(
                restart_strategy,
                min_beta,
                s_greedy,
                xi_restart,
            )
            self.restart_strategy = restart_strategy
            self.min_beta = min_beta
            self.s_greedy = s_greedy
            self.xi_restart = xi_restart

        else:
            message = "Restarting strategy must be one of {0}."
            raise ValueError(
                message.format(
                    ", ".join(self._restarting_strategies),
                ),
            )
        self._t_now = 1.0
        self._t_prev = 1.0
        self._delta0 = None
        self._safeguard = False

    def _check_restart_params(
        self,
        restart_strategy,
        min_beta,
        s_greedy,
        xi_restart,
    ):
        r"""Check restarting parameters.

        This method checks that the restarting parameters are set and satisfy
        the correct assumptions. It also checks that the current mode is
        regular (as opposed to CD for now).

        Parameters
        ----------
        restart_strategy: str or None
            Name of the restarting strategy, if ``None``, there is no
            restarting (default is ``None``)
        min_beta: float or None
            The minimum :math:`\beta` value when using the greedy restarting
            strategy (default is ``None``)
        s_greedy: float or None
            Parameter for the safeguard comparison in the greedy restarting
            strategy, it must be > 1 (default is ``None``)
        xi_restart: float or None
            Mutlitplicative parameter for the update of beta in the greedy
            restarting strategy and for the update of r_lazy in the adaptive
            restarting strategies, it must be > 1 (default is ``None``)

        Returns
        -------
        bool
            ``True``

        Raises
        ------
        ValueError
            When a parameter that should be set isn't or doesn't verify the
            correct assumptions.

        """
        if restart_strategy is None:
            return True

        if self.mode != "regular":
            raise ValueError(
                "Restarting strategies can only be used with regular mode.",
            )

        greedy_params_check = min_beta is None or s_greedy is None or s_greedy <= 1

        if restart_strategy == "greedy" and greedy_params_check:
            raise ValueError(
                "You need a min_beta and an s_greedy > 1 for greedy restart.",
            )

        if xi_restart is None or xi_restart >= 1:
            raise ValueError("You need a xi_restart < 1 for restart.")

        return True

    def is_restart(self, z_old, x_new, x_old):
        r"""Check whether the algorithm needs to restart.

        This method implements the checks necessary to tell whether the
        algorithm needs to restart depending on the restarting strategy.
        It also updates the FISTA parameters according to the restarting
        strategy (namely :math:`\beta` and :math:`r`).

        Parameters
        ----------
        z_old: numpy.ndarray
            Corresponds to :math:`y_n` in :cite:`liang2018`.
        x_new: numpy.ndarray
            Corresponds to :math:`x_{n+1}`` in :cite:`liang2018`.
        x_old: numpy.ndarray
            Corresponds to :math:`x_n` in :cite:`liang2018`.

        Returns
        -------
        bool
            Whether the algorithm should restart

        Notes
        -----
        Implements restarting and safeguarding steps in algorithms 4-5 of
        :cite:`liang2018`.

        """
        xp = backend.get_array_module(x_new)

        if self.restart_strategy is None:
            return False

        criterion = xp.vdot(z_old - x_new, x_new - x_old) >= 0

        if criterion:
            if "adaptive" in self.restart_strategy:
                self.r_lazy *= self.xi_restart
            if self.restart_strategy in {"adaptive-ii", "adaptive-2"}:
                self._t_now = 1

        if self.restart_strategy == "greedy":
            cur_delta = xp.linalg.norm(x_new - x_old)
            if self._delta0 is None:
                self._delta0 = self.s_greedy * cur_delta
            else:
                self._safeguard = cur_delta >= self._delta0

        return criterion

    def update_beta(self, beta):
        r"""Update :math:`\beta`.

        This method updates :math:`\beta` only in the case of safeguarding
        (should only be done in the greedy restarting strategy).

        Parameters
        ----------
        beta: float
            The :math:`\beta` parameter

        Returns
        -------
        float
            The new value for the :math:`\beta` parameter

        """
        if self._safeguard:
            beta *= self.xi_restart
            beta = max(beta, self.min_beta)

        return beta

    def update_lambda(self, *args, **kwargs):
        r"""Update :math:`\lambda`.

        This method updates the value of :math:`\lambda`.

        Parameters
        ----------
        *args : tuple
            Positional arguments
        **kwargs : dict
            Keyword arguments

        Returns
        -------
        float
            Current :math:`\lambda` value

        Notes
        -----
        Implements steps 3 and 4 from algoritm 10.7 in :cite:`bauschke2009`.

        """
        if self.restart_strategy == "greedy":
            return 2

        # Steps 3 and 4 from alg.10.7.
        self._t_prev = self._t_now

        if self.mode == "regular":
            sqrt_part = self.r_lazy * self._t_prev**2 + self.q_lazy
            self._t_now = self.p_lazy + np.sqrt(sqrt_part) * 0.5

        elif self.mode == "CD":
            self._t_now = (self._n + self.a_cd - 1) / self.a_cd
            self._n += 1

        return 1 + (self._t_prev - 1) / self._t_now


class ForwardBackward(SetUp):
    r"""Forward-Backward optimisation.

    This class implements standard forward-backward optimisation with an the
    option to use the FISTA speed-up.

    Parameters
    ----------
    x : numpy.ndarray
        Initial guess for the primal variable
    grad
        Gradient operator class instance
    prox
        Proximity operator class instance
    cost : class instance or str, optional
        Cost function class instance (default is ``'auto'``); Use ``'auto'`` to
        automatically generate a ``costObj`` instance
    beta_param : float, optional
        Initial value of the beta parameter, :math:`\beta` (default is ``1.0``)
    lambda_param : float, optional
        Initial value of the lambda parameter, :math:`\lambda`
        (default is ```1.0``)
    beta_update : callable, optional
        Beta parameter update method (default is ``None``)
    lambda_update : callable or str, optional
        Lambda parameter update method (default is 'fista')
    auto_iterate : bool, optional
        Option to automatically begin iterations upon initialisation (default
        is ``True``)

    Notes
    -----
    The ``beta_param`` can also be set using the keyword ``step_size``, which
    will override the value of ``beta_param``.

    The following state variable are available for metrics measurememts at
    each iteration :

    * ``'x_new'`` : new estimate of :math:`x`
    * ``'z_new'`` : new estimate of :math:`z` (adjoint representation of
      :math:`x`).
    * ``'idx'`` : index of the iteration.

    See Also
    --------
    FISTA : complementary class
    modopt.opt.algorithms.base.SetUp : parent class
    modopt.opt.cost.costObj : cost object class
    modopt.opt.gradient : gradient operator classes
    modopt.opt.proximity : proximity operator classes

    """

    def __init__(
        self,
        x,
        grad,
        prox,
        cost="auto",
        beta_param=1.0,
        lambda_param=1.0,
        beta_update=None,
        lambda_update="fista",
        auto_iterate=True,
        metric_call_period=5,
        metrics=None,
        linear=None,
        **kwargs,
    ):
        # Set default algorithm properties
        super().__init__(
            metric_call_period=metric_call_period,
            metrics=metrics,
            **kwargs,
        )

        # Set the initial variable values
        self._check_input_data(x)
        self._x_old = self.copy_data(x)
        self._z_old = self.copy_data(x)

        # Set the algorithm operators
        for operator in (grad, prox, cost):
            self._check_operator(operator)

        self._grad = grad
        self._prox = prox
        self._linear = linear

        if cost == "auto":
            self._cost_func = costObj([self._grad, self._prox])
        else:
            self._cost_func = cost

        # Check if there is a linear op, needed for metrics in the FB algoritm
        if metrics and self._linear is None:
            raise ValueError(
                "When using metrics, you must pass a linear operator",
            )

        if self._linear is None:
            self._linear = Identity()

        # Set the algorithm parameters
        for param_val in (beta_param, lambda_param):
            self._check_param(param_val)

        self._beta = self.step_size or beta_param
        self._lambda = lambda_param

        # Set the algorithm parameter update methods
        self._check_param_update(beta_update)
        self._beta_update = beta_update
        if isinstance(lambda_update, str) and lambda_update == "fista":
            fista = FISTA(**kwargs)
            self._lambda_update = fista.update_lambda
            self._is_restart = fista.is_restart
            self._beta_update = fista.update_beta
        else:
            self._check_param_update(lambda_update)
            self._lambda_update = lambda_update
            self._is_restart = lambda *args, **kwargs: False

        # Automatically run the algorithm
        if auto_iterate:
            self.iterate()

    def _update_param(self):
        """Update parameters.

        This method updates the values of the algorthm parameters with the
        methods provided.

        """
        # Update the gamma parameter.
        if not isinstance(self._beta_update, type(None)):
            self._beta = self._beta_update(self._beta)

        # Update lambda parameter.
        if not isinstance(self._lambda_update, type(None)):
            self._lambda = self._lambda_update(self._lambda)

    def _update(self):
        """Update.

        This method updates the current reconstruction.

        Notes
        -----
        Implements algorithm 10.7 (or 10.5) from :cite:`bauschke2009`.

        """
        # Step 1 from alg.10.7.
        self._grad.get_grad(self._z_old)
        y_old = self._z_old - self._beta * self._grad.grad

        # Step 2 from alg.10.7.
        self._x_new = self._prox.op(y_old, extra_factor=self._beta)

        # Step 5 from alg.10.7.
        self._z_new = self._x_old + self._lambda * (self._x_new - self._x_old)

        # Restarting step from alg.4-5 in [L2018]
        if self._is_restart(self._z_old, self._x_new, self._x_old):
            self._z_new = self._x_new

        # Update old values for next iteration.
        self._x_old = self.xp.copy(self._x_new)
        self._z_old = self.xp.copy(self._z_new)

        # Update parameter values for next iteration.
        self._update_param()

        # Test cost function for convergence.
        if self._cost_func:
            self.converge = self.any_convergence_flag() or self._cost_func.get_cost(
                self._x_new
            )

    def iterate(self, max_iter=150, progbar=None):
        """Iterate.

        This method calls update until either the convergence criteria is met
        or the maximum number of iterations is reached.

        Parameters
        ----------
        max_iter : int, optional
            Maximum number of iterations (default is ``150``)
        progbar: tqdm.tqdm
            Progress bar handle (default is ``None``)
        """
        self._run_alg(max_iter, progbar)

        # retrieve metrics results
        self.retrieve_outputs()
        # rename outputs as attributes
        self.x_final = self._z_new

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
            "x_new": self._linear.adj_op(self._x_new),
            "z_new": self._z_new,
            "idx": self.idx,
        }

    def retrieve_outputs(self):
        """Retireve outputs.

        Declare the outputs of the algorithms as attributes: ``x_final``,
        ``y_final``, ``metrics``.

        """
        metrics = {}
        for obs in self._observers["cv_metrics"]:
            metrics[obs.name] = obs.retrieve_metrics()
        self.metrics = metrics


class GenForwardBackward(SetUp):
    r"""Generalized Forward-Backward Algorithm.

    This class implements algorithm 1 from :cite:`raguet2011`.

    Parameters
    ----------
    x : list, tuple or numpy.ndarray
        Initial guess for the primal variable
    grad
        Gradient operator class
    prox_list : list
        List of proximity operator class instances
    cost : class instance or str, optional
        Cost function class instance (default is ``'auto'``); Use ``'auto'`` to
        automatically generate a ``costObj`` instance
    gamma_param : float, optional
        Initial value of the gamma parameter, :math:`\gamma`
        (default is ``1.0``)
    lambda_param : float, optional
        Initial value of the lambda parameter, :math:`\lambda`
        (default is ``1.0``)
    gamma_update : callable, optional
        Gamma parameter update method (default is ``None``)
    lambda_update : callable, optional
        Lambda parameter parameter update method (default is ``None``)
    weights : list, tuple or numpy.ndarray, optional
        Proximity operator weights (default is ``None``)
    auto_iterate : bool, optional
        Option to automatically begin iterations upon initialisation (default
        is ``True``)

    Notes
    -----
    The ``gamma_param`` can also be set using the keyword ``step_size``, which
    will override the value of ``gamma_param``.

    The following state variable are available for metrics measurememts at
    each iteration :

    * ``'x_new'`` : new estimate of :math:`x`
    * ``'z_new'`` : new estimate of :math:`z` (adjoint representation of
      :math:`x`).
    * ``'idx'`` : index of the iteration.

    See Also
    --------
    modopt.opt.algorithms.base.SetUp : parent class
    modopt.opt.cost.costObj : cost object class
    modopt.opt.gradient : gradient operator classes
    modopt.opt.proximity : proximity operator classes

    """

    def __init__(
        self,
        x,
        grad,
        prox_list,
        cost="auto",
        gamma_param=1.0,
        lambda_param=1.0,
        gamma_update=None,
        lambda_update=None,
        weights=None,
        auto_iterate=True,
        metric_call_period=5,
        metrics=None,
        linear=None,
        **kwargs,
    ):
        # Set default algorithm properties
        super().__init__(
            metric_call_period=metric_call_period,
            metrics=metrics,
            **kwargs,
        )

        # Set the initial variable values
        self._check_input_data(x)
        self._x_old = self.xp.copy(x)

        # Set the algorithm operators
        for operator in [grad, cost, *prox_list]:
            self._check_operator(operator)

        self._grad = grad
        self._prox_list = self.xp.array(prox_list)
        self._linear = linear

        if cost == "auto":
            self._cost_func = costObj([self._grad, *prox_list])
        else:
            self._cost_func = cost

        # Check if there is a linear op, needed for metrics in the FB algoritm
        if metrics and self._linear is None:
            raise ValueError(
                "When using metrics, you must pass a linear operator",
            )

        if self._linear is None:
            self._linear = Identity()

        # Set the algorithm parameters
        for param_val in (gamma_param, lambda_param):
            self._check_param(param_val)

        self._gamma = self.step_size or gamma_param
        self._lambda_param = lambda_param

        # Set the algorithm parameter update methods
        for param_update in (gamma_update, lambda_update):
            self._check_param_update(param_update)

        self._gamma_update = gamma_update
        self._lambda_update = lambda_update

        # Set the proximity weights
        self._set_weights(weights)

        # Set initial z
        self._z = self.xp.array([self._x_old for i in range(self._prox_list.size)])

        # Automatically run the algorithm
        if auto_iterate:
            self.iterate()

    def _set_weights(self, weights):
        """Set weights.

        This method sets weights on each of the proximty operators provided.

        Parameters
        ----------
        weights : list, tuple or numpy.ndarray
            List of weights

        Raises
        ------
        TypeError
            For invalid input type
        ValueError
            If weights do not sum to one

        """
        if isinstance(weights, type(None)):
            weights = self.xp.repeat(
                1.0 / self._prox_list.size,
                self._prox_list.size,
            )
        elif not isinstance(weights, (list, tuple, np.ndarray)):
            raise TypeError("Weights must be provided as a list.")

        weights = self.xp.array(weights)

        if not np.issubdtype(weights.dtype, np.floating):
            raise ValueError("Weights must be list of float values.")

        if weights.size != self._prox_list.size:
            raise ValueError(
                "The number of weights must match the number of proximity "
                + "operators.",
            )

        expected_weight_sum = 1.0

        if self.xp.sum(weights) != expected_weight_sum:
            raise ValueError(
                "Proximity operator weights must sum to 1.0. Current sum of "
                + f"weights = {self.xp.sum(weights)}",
            )

        self._weights = weights

    def _update_param(self):
        """Update parameters.

        This method updates the values of the algorthm parameters with the
        methods provided.

        """
        # Update the gamma parameter.
        if not isinstance(self._gamma_update, type(None)):
            self._gamma = self._gamma_update(self._gamma)

        # Update lambda parameter.
        if not isinstance(self._lambda_update, type(None)):
            self._lambda_param = self._lambda_update(self._lambda_param)

    def _update(self):
        """Update.

        This method updates the current reconstruction.

        Notes
        -----
        Implements algorithm 1 from :cite:`raguet2011`.

        """
        # Calculate gradient for current iteration.
        self._grad.get_grad(self._x_old)

        # Update z values.
        for i in range(self._prox_list.size):
            z_temp = 2 * self._x_old - self._z[i] - self._gamma * self._grad.grad
            z_prox = self._prox_list[i].op(
                z_temp,
                extra_factor=self._gamma / self._weights[i],
            )
            self._z[i] += self._lambda_param * (z_prox - self._x_old)

        # Update current reconstruction.
        self._x_new = self.xp.sum(
            [z_i * w_i for z_i, w_i in zip(self._z, self._weights)],
            axis=0,
        )

        # Update old values for next iteration.
        self.xp.copyto(self._x_old, self._x_new)

        # Update parameter values for next iteration.
        self._update_param()

        # Test cost function for convergence.
        if self._cost_func:
            self.converge = self._cost_func.get_cost(self._x_new)

    def iterate(self, max_iter=150, progbar=None):
        """Iterate.

        This method calls update until either convergence criteria is met or
        the maximum number of iterations is reached.

        Parameters
        ----------
        max_iter : int, optional
            Maximum number of iterations (default is ``150``)
        progbar: tqdm.tqdm
            Progress bar handle (default is ``None``)
        """
        self._run_alg(max_iter, progbar)

        # retrieve metrics results
        self.retrieve_outputs()

        self.x_final = self._x_new

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
            "x_new": self._linear.adj_op(self._x_new),
            "z_new": self._z,
            "idx": self.idx,
        }

    def retrieve_outputs(self):
        """Retrieve outputs.

        Declare the outputs of the algorithms as attributes: ``x_final``,
        ``y_final``, ``metrics``.

        """
        metrics = {}
        for obs in self._observers["cv_metrics"]:
            metrics[obs.name] = obs.retrieve_metrics()
        self.metrics = metrics


class POGM(SetUp):
    r"""Proximal Optimised Gradient Method.

    This class implements algorithm 3 from :cite:`kim2017`.

    Parameters
    ----------
    u : numpy.ndarray
        Initial guess for the :math:`u` variable
    x : numpy.ndarray
        Initial guess for the :math:`x` variable (primal)
    y : numpy.ndarray
        Initial guess for the :math:`y` variable
    z : numpy.ndarray
        Initial guess for the :math:`z` variable
    grad : GradBasic
        Gradient operator class
    prox : ProximalParent
        Proximity operator class
    cost : class instance or str, optional
        Cost function class instance (default is ``'auto'``); Use ``'auto'`` to
        automatically generate a ``costObj`` instance
    linear : class instance, optional
        Linear operator class instance (default is ``None``)
    beta_param : float, optional
        Initial value of the beta parameter, :math:`\beta` (default is ``1.0``)
        This corresponds to (1 / L) in :cite:`kim2017`
    sigma_bar : float, optional
        Value of the shrinking parameter, :math:`\bar{\sigma}`
        (default is ``1.0``)
    auto_iterate : bool, optional
        Option to automatically begin iterations upon initialisation (default
        is ``True``)

    Notes
    -----
    The ``beta_param`` can also be set using the keyword ``step_size``, which
    will override the value of ``beta_param``.

    The following state variable are available for metrics measurememts at
    each iterations:

    * ``'u_new'`` : new estimate of :math:`u`
    * ``'x_new'`` : new estimate of :math:`x`
    * ``'y_new'`` : new estimate of :math:`y`
    * ``'z_new'`` : new estimate of :math:`z`
    * ``'xi'``: :math:`\xi` variable
    * ``'t'`` : new estimate of :math:`t`
    * ``'sigma'``: :math:`\sigma` variable
    * ``'idx'`` : index of the iteration.

    See Also
    --------
    modopt.opt.algorithms.base.SetUp : parent class
    modopt.opt.cost.costObj : cost object class
    modopt.opt.gradient : gradient operator classes
    modopt.opt.proximity : proximity operator classes
    modopt.opt.linear : linear operator classes

    """

    def __init__(
        self,
        u,
        x,
        y,
        z,
        grad,
        prox,
        cost="auto",
        linear=None,
        beta_param=1.0,
        sigma_bar=1.0,
        auto_iterate=True,
        metric_call_period=5,
        metrics=None,
        **kwargs,
    ):
        # Set default algorithm properties
        super().__init__(
            metric_call_period=metric_call_period,
            metrics=metrics,
            linear=linear,
            **kwargs,
        )

        # set the initial variable values
        for input_data in (u, x, y, z):
            self._check_input_data(input_data)

        self._u_old = self.xp.copy(u)
        self._x_old = self.xp.copy(x)
        self._y_old = self.xp.copy(y)
        self._z = self.xp.copy(z)

        # Set the algorithm operators
        for operator in (grad, prox, cost):
            self._check_operator(operator)

        self._grad = grad
        self._prox = prox
        self._linear = linear
        if cost == "auto":
            self._cost_func = costObj([self._grad, self._prox])
        else:
            self._cost_func = cost

        # If linear is None, make it Identity for call of metrics
        if self._linear is None:
            self._linear = Identity()

        # Set the algorithm parameters
        for param_val in (beta_param, sigma_bar):
            self._check_param(param_val)
        if sigma_bar < 0 or sigma_bar > 1:
            raise ValueError("The sigma bar parameter needs to be in [0, 1]")

        self._beta = self.step_size or beta_param
        self._sigma_bar = sigma_bar
        self._xi = 1.0
        self._sigma = 1.0
        self._t_old = 1.0
        self._grad.get_grad(self._x_old)
        self._g_old = self._grad.grad

        # Automatically run the algorithm
        if auto_iterate:
            self.iterate()

    def _update(self):
        """Update.

        This method updates the current reconstruction.

        Notes
        -----
        Implements algorithm 3 from :cite:`kim2017`.

        """
        # Step 4 from alg. 3
        self._grad.get_grad(self._x_old)
        # self._u_new = self._x_old - self._beta * self._grad.grad
        self._u_new = -self._beta * self._grad.grad
        self._u_new += self._x_old

        # Step 5 from alg. 3
        self._t_new = 0.5 * (1 + self.xp.sqrt(1 + 4 * self._t_old**2))

        # Step 6 from alg. 3
        t_shifted_ratio = (self._t_old - 1) / self._t_new
        sigma_t_ratio = self._sigma * self._t_old / self._t_new
        beta_xi_t_shifted_ratio = t_shifted_ratio * self._beta / self._xi
        self._z = -beta_xi_t_shifted_ratio * (self._x_old - self._z)
        self._z += self._u_new
        self._z += t_shifted_ratio * (self._u_new - self._u_old)
        self._z += sigma_t_ratio * (self._u_new - self._x_old)

        # Step 7 from alg. 3
        self._xi = self._beta * (1 + t_shifted_ratio + sigma_t_ratio)

        # Step 8 from alg. 3
        self._x_new = self._prox.op(self._z, extra_factor=self._xi)

        # Restarting and gamma-Decreasing
        # Step 9 from alg. 3
        # self._g_new = self._grad.grad - (self._x_new - self._z) / self._xi
        self._g_new = self._z - self._x_new
        self._g_new /= self._xi
        self._g_new += self._grad.grad

        # Step 10 from alg 3.
        # self._y_new = self._x_old - self._beta * self._g_new
        self._y_new = -self._beta * self._g_new
        self._y_new += self._x_old

        # Step 11 from alg. 3
        restart_crit = self.xp.vdot(-self._g_new, self._y_new - self._y_old) < 0
        if restart_crit:
            self._t_new = 1
            self._sigma = 1

        # Step 13 from alg. 3
        elif self.xp.vdot(self._g_new, self._g_old) < 0:
            self._sigma *= self._sigma_bar

        # updating variables
        self._t_old = self._t_new
        self.xp.copyto(self._u_old, self._u_new)
        self.xp.copyto(self._x_old, self._x_new)
        self.xp.copyto(self._g_old, self._g_new)
        self.xp.copyto(self._y_old, self._y_new)

        # Test cost function for convergence.
        if self._cost_func:
            self.converge = self.any_convergence_flag() or self._cost_func.get_cost(
                self._x_new
            )

    def iterate(self, max_iter=150, progbar=None):
        """Iterate.

        This method calls update until either convergence criteria is met or
        the maximum number of iterations is reached.

        Parameters
        ----------
        max_iter : int, optional
            Maximum number of iterations (default is ``150``)
        progbar: tqdm.tqdm
            Progress bar handle (default is ``None``)
        """
        self._run_alg(max_iter, progbar)

        # retrieve metrics results
        self.retrieve_outputs()
        # rename outputs as attributes
        self.x_final = self._x_new

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
            "u_new": self._u_new,
            "x_new": self._linear.adj_op(self._x_new),
            "y_new": self._y_new,
            "z_new": self._z,
            "xi": self._xi,
            "sigma": self._sigma,
            "t": self._t_new,
            "idx": self.idx,
        }

    def retrieve_outputs(self):
        """Retrieve outputs.

        Declare the outputs of the algorithms as attributes: ``x_final``,
        ``y_final``, ``metrics``.

        """
        metrics = {}
        for obs in self._observers["cv_metrics"]:
            metrics[obs.name] = obs.retrieve_metrics()
        self.metrics = metrics
