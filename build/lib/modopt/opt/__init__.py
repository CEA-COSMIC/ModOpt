# -*- coding: utf-8 -*-

"""OPTIMISATION PROBLEM MODULES

This module contains submodules for solving optimisation problems.

:Author: Samuel Farrens <samuel.farrens@cea.fr>

References
----------

.. [Con2013] Condat, A Primal-Dual Splitting Method for Convex Optimization
    Involving Lipschitzian, Proximable and Linear Composite Terms, 2013,
    Journal of Optimization Theory and Applications, 158, 2, 460.
    [https://link.springer.com/article/10.1007/s10957-012-0245-9]

.. [B2011] Bauschke et al., Fixed-Point Algorithms for Inverse Problems in
    Science and Engineering, 2011, Chapter 10.
    [http://rentals.springer.com/product/9781441995698]

.. [R2012] Raguet et al., Generalized Forward-Backward Splitting, 2012, SIAM,
    v6. [https://arxiv.org/abs/1108.4404]

.. [CWB2007] Candes, Wakin and Boyd, Enhancing Sparsity by Reweighting l1
    Minimization, 2007, Journal of Fourier Analysis and Applications,
    14(5):877-905. [https://arxiv.org/abs/0711.1612]

"""

__all__ = ['cost', 'gradient', 'linear', 'algorithms', 'proximity', 'reweight']

from . import *
