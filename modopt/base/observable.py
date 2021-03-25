# -*- coding: utf-8 -*-

"""Observable.

This module contains observable classes

:Author: Benoir Sarthou

"""

import time

import numpy as np


class SignalObject(object):
    """Dummy class for signals."""

    pass


class Observable(object):
    """Base class for observable classes.

    This class defines a simple interface to add or remove observers
    on an object.

    Parameters
    ----------
    signals : list
        The allowed signals

    """

    def __init__(self, signals):

        # Define class parameters
        self._allowed_signals = []
        self._observers = {}

        # Set allowed signals
        for signal in signals:
            self._allowed_signals.append(signal)
            self._observers[signal] = []

        # Set a lock option to avoid multiple observer notifications
        self._locked = False

    def add_observer(self, signal, observer):
        """Add an observer to the object.

        Raise an exception if the signal is not allowed.

        Parameters
        ----------
        signal : str
            A valid signal
        observer : callable
            A function that will be called when the signal is emitted

        """
        self._is_allowed_signal(signal)
        self._add_observer(signal, observer)

    def remove_observer(self, signal, observer):
        """Remove an observer from the object.

        Raise an eception if the signal is not allowed.

        Parameters
        ----------
        signal : str
            A valid signal
        observer : callable
            An obervation function to be removed

        """
        self._is_allowed_event(signal)
        self._remove_observer(signal, observer)

    def notify_observers(self, signal, **kwargs):
        """Notify observers of a given signal.

        Parameters
        ----------
        signal : str
            A valid signal
        kwargs : dict
            The parameters that will be sent to the observers

        Returns
        -------
        bool
            ``False`` if a notification is in progress, otherwise ``True``

        """
        # Check if a notification if in progress
        if self._locked:
            return False
        # Set the lock
        self._locked = True

        # Create a signal object
        signal_to_be_notified = SignalObject()
        signal_to_be_notified.object = self
        signal_to_be_notified.signal = signal

        for name, key_value in kwargs.items():
            setattr(signal_to_be_notified, name, key_value)
        # Notify all the observers
        for observer in self._observers[signal]:
            observer(signal_to_be_notified)
        # Unlock the notification process
        self._locked = False

    def _get_allowed_signals(self):
        """Get allowed signals.

        Events allowed for the current object.

        Returns
        -------
        list
            List of allowed signals

        """
        return self._allowed_signals

    allowed_signals = property(_get_allowed_signals)

    def _is_allowed_signal(self, signal):
        """Check if a signal is valid.

        Raise an exception if the signal is not allowed.

        Parameters
        ----------
        signal: str
            A signal

        Raises
        ------
        ValueError
            For invalid signal

        """
        if signal not in self._allowed_signals:
            message = 'Signal "{0}" is not allowed for "{1}"'
            raise ValueError(message.format(signal, type(self)))

    def _add_observer(self, signal, observer):
        """Associate an observer to a valid signal.

        Parameters
        ----------
        signal : str
            A valid signal
        observer : callable
            An obervation function

        """
        if observer not in self._observers[signal]:
            self._observers[signal].append(observer)

    def _remove_observer(self, signal, observer):
        """Remove an observer to a valid signal.

        Parameters
        ----------
        signal : str
            A valid signal
        observer : callable
            An obervation function to be removed

        """
        if observer in self._observers[signal]:
            self._observers[signal].remove(observer)


class MetricObserver(object):
    """Metric observer.

    Wrapper of the metric to the observer object notify by the Observable
    class.

    Parameters
    ----------
    name : str
        The name of the metric
    metric : callable
        Metric function with this precise signature func(test, ref)
    mapping : dict
        Define the mapping between the iterate variable and the metric
        keyword: {'x_new':'name_var_1', 'y_new':'name_var_2'}. To cancel
        the need of a variable, the dict value should be None:
        'y_new':None.
    cst_kwargs : dict
        Keywords arguments of constant argument for the metric computation
    early_stopping : bool
        If True it will compute the convergence flag (default is ``False``)
    wind : int
        Window on with the convergence criteria is compute (default is ``6``)
    eps : float
        The level of criteria of convergence (default is ``1.0e-3``)

    """

    def __init__(
        self,
        name,
        metric,
        mapping,
        cst_kwargs,
        early_stopping=False,
        wind=6,
        eps=1.0e-3,
    ):

        self.name = name
        self.metric = metric
        self.mapping = mapping
        self.cst_kwargs = cst_kwargs
        self.list_cv_values = []
        self.list_iters = []
        self.list_dates = []
        self.eps = eps
        self.wind = wind
        self.converge_flag = False
        self.early_stopping = early_stopping

    def __call__(self, signal):
        """Call Method.

        Wrapper the call from the observer signature to the metric
        signature.

        Parameters
        ----------
        signal : str
            A valid signal

        """
        kwargs = {}
        for key, key_value in self.mapping.items():
            if key_value is not None:
                kwargs[key_value] = getattr(signal, key)
        kwargs.update(self.cst_kwargs)
        self.list_iters.append(signal.idx)
        self.list_dates.append(time.time())
        self.list_cv_values.append(self.metric(**kwargs))

        if self.early_stopping:
            self.is_converge()

    def is_converge(self):
        """Check convergence.

        Return ``True`` if the convergence criteria is matched.

        """
        if len(self.list_cv_values) < self.wind:
            return
        start_idx = -self.wind
        mid_idx = -(self.wind // 2)
        old_mean = np.array(self.list_cv_values[start_idx:mid_idx]).mean()
        current_mean = np.array(self.list_cv_values[mid_idx:]).mean()
        normalize_residual_metrics = (
            np.abs(old_mean - current_mean) / np.abs(old_mean)
        )
        self.converge_flag = normalize_residual_metrics < self.eps

    def retrieve_metrics(self):
        """Retrieve metrics.

        Return the convergence metrics saved with the corresponding
        iterations.

        Returns
        -------
        dict
            Convergence metrics

        """
        time_val = np.array(self.list_dates)

        if time_val.size >= 1:
            time_val -= time_val[0]

        return {
            'time': time_val,
            'index': self.list_iters,
            'values': self.list_cv_values,
        }
