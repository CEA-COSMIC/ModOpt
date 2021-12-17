# -*- coding: utf-8 -*-
"""Base SetUp for optimisation algorithms."""

from inspect import getmro

import numpy as np
from progressbar import ProgressBar

from modopt.base import backend
from modopt.base.observable import MetricObserver, Observable
from modopt.interface.errors import warn


class SetUp(Observable):
    r"""Algorithm Set-Up.

    This class contains methods for checking the set-up of an optimisation
    algotithm and produces warnings if they do not comply.

    Parameters
    ----------
    metric_call_period : int, optional
        Metric call period (default is ``5``)
    metrics : dict, optional
        Metrics to be used (default is ``\{\}``)
    verbose : bool, optional
        Option for verbose output (default is ``False``)
    progress : bool, optional
        Option to display progress bar (default is ``True``)
    step_size : int, optional
        Generic step size parameter to override default algorithm
        parameter name (`e.g.` `step_size` will override the value set for
        `beta_param` in `ForwardBackward`)
    use_gpu : bool, optional
        Option to use available GPU

    See Also
    --------
    modopt.base.observable.Observable : parent class
    modopt.base.observable.MetricObserver : definition of metrics

    """

    def __init__(
        self,
        metric_call_period=5,
        metrics=None,
        verbose=False,
        progress=True,
        step_size=None,
        compute_backend='numpy',
        **dummy_kwargs,
    ):
        self.idx = 0
        self.converge = False
        self.verbose = verbose
        self.progress = progress
        self.metrics = metrics
        self.step_size = step_size
        self._op_parents = (
            'GradParent',
            'ProximityParent',
            'LinearParent',
            'costObj',
        )

        self.metric_call_period = metric_call_period

        # Declaration of observers for metrics
        super().__init__(['cv_metrics'])

        for name, dic in self.metrics.items():
            observer = MetricObserver(
                name,
                dic['metric'],
                dic['mapping'],
                dic['cst_kwargs'],
                dic['early_stopping'],
            )
            self.add_observer('cv_metrics', observer)

        xp, compute_backend = backend.get_backend(compute_backend)
        self.xp = xp
        self.compute_backend = compute_backend

    @property
    def metrics(self):
        """Set metrics dictionary."""
        return self._metrics

    @metrics.setter
    def metrics(self, metrics):

        if isinstance(metrics, type(None)):
            self._metrics = {}
        elif isinstance(metrics, dict):
            self._metrics = metrics
        else:
            raise TypeError(
                'Metrics must be a dictionary, not {0}.'.format(type(metrics)),
            )

    def any_convergence_flag(self):
        """Check convergence flag.

        Retur True if any matrix values matched the convergence criteria.

        Returns
        -------
        bool
            True if any convergence criteria met

        """
        return any(
            obs.converge_flag for obs in self._observers['cv_metrics']
        )

    def copy_data(self, input_data):
        """Copy Data.

        Set directive for copying data.

        Parameters
        ----------
        input_data : numpy.ndarray
            Input data

        Returns
        -------
        numpy.ndarray
            Copy of input data

        """
        return self.xp.copy(backend.change_backend(
            input_data,
            self.compute_backend,
        ))

    def _check_input_data(self, input_data):
        """Check input data type.

        This method checks if the input data is a numpy array

        Parameters
        ----------
        input_data : numpy.ndarray
            Input data array

        Raises
        ------
        TypeError
            For invalid input type

        """
        if not (isinstance(input_data, (self.xp.ndarray, np.ndarray))):
            raise TypeError(
                'Input data must be a numpy array or backend array',
            )

    def _check_param(self, param_val):
        """Check algorithm parameters.

        This method checks if the specified algorithm parameters are floats

        Parameters
        ----------
        param_val : float
            Parameter value

        Raises
        ------
        TypeError
            For invalid input type

        """
        if not isinstance(param_val, float):
            raise TypeError('Algorithm parameter must be a float value.')

    def _check_param_update(self, param_update):
        """Check algorithm parameter update methods.

        This method checks if the specified algorithm parameters are floats

        Parameters
        ----------
        param_update : callable
            Callable function

        Raises
        ------
        TypeError
            For invalid input type

        """
        param_conditions = (
            not isinstance(param_update, type(None))
            and not callable(param_update)
        )

        if param_conditions:
            raise TypeError(
                'Algorithm parameter update must be a callabale function.',
            )

    def _check_operator(self, operator):
        """Check set-Up.

        This method checks algorithm operator against the expected parent
        classes

        Parameters
        ----------
        operator : str
            Algorithm operator to check

        """
        if not isinstance(operator, type(None)):
            tree = [op_obj.__name__ for op_obj in getmro(operator.__class__)]

            if not any(parent in tree for parent in self._op_parents):
                message = '{0} does not inherit an operator parent.'
                warn(message.format(str(operator.__class__)))

    def _compute_metrics(self):
        """Compute metrics during iteration.

        This method create the args necessary for metrics computation, then
        call the observers to compute metrics

        """
        kwargs = self.get_notify_observers_kwargs()
        self.notify_observers('cv_metrics', **kwargs)

    def _iterations(self, max_iter, progbar=None):
        """Iterate method.

        Iterate the update step of the given algorithm.

        Parameters
        ----------
        max_iter : int
            Maximum number of iterations
        progbar : progressbar.bar.ProgressBar
            Progress bar (default is ``None``)

        """
        for idx in range(max_iter):
            self.idx = idx
            self._update()

            # Calling metrics every metric_call_period cycle
            # Also calculate at the end (max_iter or at convergence)
            # We do not call metrics if metrics is empty or metric call
            # period is None
            if self.metrics and self.metric_call_period is not None:

                metric_conditions = (
                    self.idx % self.metric_call_period == 0
                    or self.idx == (max_iter - 1)
                    or self.converge,
                )

                if metric_conditions:
                    self._compute_metrics()

            if self.converge:
                if self.verbose:
                    print(' - Converged!')
                break

            if not isinstance(progbar, type(None)):
                progbar.update(idx)

    def _run_alg(self, max_iter):
        """Run algorithm.

        Run the update step of a given algorithm up to the maximum number of
        iterations.

        Parameters
        ----------
        max_iter : int
            Maximum number of iterations

        See Also
        --------
        progressbar.bar.ProgressBar

        """
        if self.progress:
            with ProgressBar(
                redirect_stdout=True,
                max_value=max_iter,
            ) as progbar:
                self._iterations(max_iter, progbar=progbar)
        else:
            self._iterations(max_iter)
