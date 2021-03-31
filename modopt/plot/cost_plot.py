# -*- coding: utf-8 -*-

"""PLOTTING ROUTINES.

This module contains methods for making plots.

:Author: Samuel Farrens <samuel.farrens@cea.fr>

"""

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    import_fail = True
else:
    import_fail = False


def plotCost(cost_list, output=None):
    """Plot cost function.

    Plot the final cost function.

    Parameters
    ----------
    cost_list : list
        List of cost function values
    output : str, optional
        Output file name (default is ``None``)

    Raises
    ------
    ImportError
        If Matplotlib package not found

    """
    if import_fail:
        raise ImportError('Matplotlib package not found')

    else:
        if isinstance(output, type(None)):
            file_name = 'cost_function.png'
        else:
            file_name = '{0}_cost_function.png'.format(output)

        plt.figure()
        plt.plot(np.log10(cost_list), 'r-')
        plt.title('Cost Function')
        plt.xlabel('Iteration')
        plt.ylabel(r'$\log_{10}$ Cost')
        plt.savefig(file_name)
        plt.close()

        print(' - Saving cost function data to:', file_name)
