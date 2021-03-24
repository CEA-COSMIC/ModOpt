# -*- coding: utf-8 -*-

"""COST FUNCTIONS.

This module contains classes of different cost functions for optimization.

:Author: Samuel Farrens <samuel.farrens@cea.fr>

"""

import numpy as np

from modopt.base.backend import get_array_module
from modopt.base.types import check_callable
from modopt.plot.cost_plot import plotCost


class costObj(object):
    """Generic cost function object.

    This class updates the cost according to the input cost functio class and
    tests for convergence

    Parameters
    ----------
    costFunc : class
        Class for calculating the cost
    initial_cost : float, optional
        Initial value of the cost (default is ``1e6``)
    tolerance : float, optional
        Tolerance threshold for convergence (default is ``1e-4``)
    cost_interval : int, optional
        Iteration interval to calculate cost (default is ``1``).
        If `cost_interval` is ``None`` the cost is never calculated,
        thereby saving on computation time.
    test_range : int, optional
        Number of cost values to be used in test (default is ``4``)
    verbose : bool, optional
        Option for verbose output (default is ``True``)
    plot_output : str, optional
        Output file name for cost function plot

    Notes
    -----
    The costFunc class must contain a method called `calc_cost()`.

    Examples
    --------
    >>> from modopt.opt.cost import *
    >>> class dummy(object):
    ...     def cost(self, x):
    ...         return x ** 2
    ...
    ...
    >>> inst = costObj([dummy(), dummy()])
    >>> inst.get_cost(2)
     - ITERATION: 1
     - COST: 8
    <BLANKLINE>
    False
    >>> inst.get_cost(2)
     - ITERATION: 2
     - COST: 8
    <BLANKLINE>
    False
    >>> inst.get_cost(2)
     - ITERATION: 3
     - COST: 8
    <BLANKLINE>
    False

    """

    def __init__(
        self,
        operators,
        initial_cost=1e6,
        tolerance=1e-4,
        cost_interval=1,
        test_range=4,
        verbose=True,
        plot_output=None,
    ):

        self._operators = operators
        if not isinstance(operators, type(None)):
            self._check_operators()
        self.cost = initial_cost
        self._cost_list = []
        self._cost_interval = cost_interval
        self._iteration = 1
        self._test_list = []
        self._test_range = test_range
        self._tolerance = tolerance
        self._plot_output = plot_output
        self._verbose = verbose

    def _check_operators(self):
        """Check Operators.

        This method checks if the input operators have a `cost` method

        Raises
        ------
        TypeError
            For invalid operators type
        ValueError
            For operators without `cost` method

        """
        if not isinstance(self._operators, (list, tuple, np.ndarray)):
            message = (
                'Input operators must be provided as a list, not {0}'
            )
            raise TypeError(message.format(type(self._operators)))

        for op in self._operators:
            if not hasattr(op, 'cost'):
                raise ValueError('Operators must contain "cost" method.')
            op.cost = check_callable(op.cost)

    def _check_cost(self):
        """Check cost function.

        This method tests the cost function for convergence in the specified
        interval of iterations using the last n (test_range) cost values

        Returns
        -------
        bool
            Result of the convergence test

        """
        # Add current cost value to the test list
        self._test_list.append(self.cost)

        xp = get_array_module(self.cost)

        # Check if enough cost values have been collected
        if len(self._test_list) == self._test_range:

            # The mean of the first half of the test list
            t1 = xp.mean(
                xp.array(self._test_list[len(self._test_list) // 2:]),
                axis=0,
            )
            # The mean of the second half of the test list
            t2 = xp.mean(
                xp.array(self._test_list[:len(self._test_list) // 2]),
                axis=0,
            )
            # Calculate the change across the test list
            if xp.around(t1, decimals=16):
                cost_diff = (xp.linalg.norm(t1 - t2) / xp.linalg.norm(t1))
            else:
                cost_diff = 0

            # Reset the test list
            self._test_list = []

            if self._verbose:
                print(' - CONVERGENCE TEST - ')
                print(' - CHANGE IN COST:', cost_diff)
                print('')

            # Check for convergence
            return cost_diff <= self._tolerance

        return False

    def _calc_cost(self, *args, **kwargs):
        """Calculate the cost.

        This method calculates the cost from each of the input operators.

        Parameters
        ----------
        args : interable
            Positional arguments
        kwargs : dict
            Keyword arguments

        Returns
        -------
        float
            Cost

        """
        return np.sum([op.cost(*args, **kwargs) for op in self._operators])

    def get_cost(self, *args, **kwargs):
        """Get cost function.

        This method calculates the current cost and tests for convergence.

        Parameters
        ----------
        args : interable
            Positional arguments
        kwargs : dict
            Keyword arguments

        Returns
        -------
        bool
            Result of the convergence test

        """
        # Check if the cost should be calculated
        test_conditions = (
            self._cost_interval is None
            or self._iteration % self._cost_interval
        )

        if test_conditions:
            test_result = False

        else:
            if self._verbose:
                print(' - ITERATION:', self._iteration)

            # Calculate the current cost
            self.cost = self._calc_cost(verbose=self._verbose, *args, **kwargs)
            self._cost_list.append(self.cost)

            if self._verbose:
                print(' - COST:', self.cost)
                print('')

            # Test for convergence
            test_result = self._check_cost()

        # Update the current iteration number
        self._iteration += 1

        return test_result

    def plot_cost(self):  # pragma: no cover
        """Plot the cost function.

        This method plots the cost function as function of iteration number

        """
        plotCost(self._cost_list, self._plot_output)
