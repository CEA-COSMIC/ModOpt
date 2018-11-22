# -*- coding: utf-8 -*-

"""PLOTTING ROUTINES

This module contains methods for making plots.

:Author: Samuel Farrens <samuel.farrens@cea.fr>

"""

from __future__ import print_function
import numpy as np
from modopt.interface.errors import warn
try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    import_fail = True
else:
    import_fail = False


def plotCost(cost_list, output=None):
    """Plot cost function

    Plot the final cost function

    Parameters
    ----------
    cost_list : list
        List of cost function values
    output : str, optional
        Output file name

    """

    if not import_fail:

        if isinstance(output, type(None)):
            file_name = 'cost_function.png'
        else:
            file_name = output + '_cost_function.png'

        plt.figure()
        plt.plot(np.log10(cost_list), 'r-')
        plt.title('Cost Function')
        plt.xlabel('Iteration')
        plt.ylabel(r'$\log_{10}$ Cost')
        plt.savefig(file_name)
        plt.close()

        print(' - Saving cost function data to:', file_name)

    else:

        warn('Matplotlib not installed.')
