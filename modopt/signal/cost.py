# -*- coding: utf-8 -*-

"""COST FUNCTIONS

This module contains classes of different cost functions for optimization.

:Author: Samuel Farrens <samuel.farrens@gmail.com>

:Version: 2.2

:Date: 23/10/2017

"""

from __future__ import division, print_function
import numpy as np
try:
    from modopt.plot.cost_plot import plotCost
except ImportError:
    import_fail = True
else:
    import_fail = False


class costObj(object):

    """Generic cost function object

    This class updates the cost according to the input cost functio class and
    tests for convergence

    Parameters
    ----------
    costFunc : class
        Class for calculating the cost
    initial_cost : float, optional
        Initial value of the cost (default is "1e6")
    tolerance : float, optional
        Tolerance threshold for convergence (default is "1e-4")
    cost_interval : int, optional
        Iteration interval to calculate cost (default is "1")
    test_range : int, optional
        Number of cost values to be used in test (default is "4")
    verbose : bool, optional
        Option for verbose output (default is "True")
    plot_output : str, optional
        Output file name for cost function plot

    Notes
    -----
    The costFunc class must contain a method called `calc_cost()`.

    """

    def __init__(self, costFunc, initial_cost=1e6, tolerance=1e-4,
                 cost_interval=1, test_range=4, verbose=True,
                 plot_output=None):

        if not hasattr(costFunc, 'calc_cost'):
            raise ValueError('costFunc must contain "calc_cost" method.')

        self.costFunc = costFunc
        self.cost = initial_cost
        self._cost_list = []
        self._cost_interval = cost_interval
        self._iteration = 1
        self._test_list = []
        self._test_range = test_range
        self._tolerance = tolerance
        self._plot_output = plot_output
        self._verbose = verbose

    def _check_cost(self):
        """Check cost function

        This method tests the cost function for convergence in the specified
        interval of iterations using the last n (test_range) cost values

        Returns
        -------
        bool result of the convergence test

        """

        # Add current cost value to the test list
        self._test_list.append(self.cost)

        # Check if enough cost values have been collected
        if len(self._test_list) == self._test_range:

            # The mean of the first half of the test list
            t1 = np.mean(self._test_list[len(self._test_list) // 2:], axis=0)
            # The mean of the second half of the test list
            t2 = np.mean(self._test_list[:len(self._test_list) // 2], axis=0)
            # Calculate the change across the test list
            cost_diff = (np.linalg.norm(t1 - t2) / np.linalg.norm(t1))
            # Reset the test list
            self._test_list = []

            if self._verbose:
                print(' - CONVERGENCE TEST - ')
                print(' - CHANGE IN COST:', cost_diff)
                print('')

            # Check for convergence
            return cost_diff <= self._tolerance

        else:

            return False

    def get_cost(self, *args, **kwargs):
        """Get cost function

        This method calculates the current cost and tests for convergence

        Returns
        -------
        bool result of the convergence test

        """

        # Check if the cost should be calculated
        if self._iteration % self._cost_interval:

            test_result = False

        else:

            if self._verbose:
                print(' - ITERATION:', self._iteration)

            # Calculate the current cost
            self.cost = self.costFunc.calc_cost(*args, **kwargs)
            self._cost_list.append(self.cost)

            if self._verbose:
                print(' - Log10 COST:', np.log10(self.cost))
                print('')

            # Test for convergence
            test_result = self._check_cost()

        # Update the current iteration number
        self._iteration += 1

        return test_result

    def plot_cost(self):
        """Plot the cost function

        This method plots the cost function as function of iteration number

        """

        if import_fail:
            pass

        else:
            plotCost(self._cost_list, self._plot_output)
