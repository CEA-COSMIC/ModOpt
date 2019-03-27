# -*- coding: utf-8 -*-

"""OPTIMISATION PROBLEM MODULES

This module contains submodules for solving optimisation problems.

:Author: Samuel Farrens <samuel.farrens@cea.fr>

References
----------
.. [K2018] Kim, D., & Fessler, J. A. (2018).
    Adaptive restart of the optimized gradient method for convex optimization.
    Journal of Optimization Theory and Applications, 178(1), 240-263.
    [https://link.springer.com/content/pdf/10.1007%2Fs10957-018-1287-4.pdf]
.. [L2018] Liang, Jingwei, and Carola-Bibiane Schönlieb.
    Improving FISTA: Faster, Smarter and Greedier.
    arXiv preprint arXiv:1811.01430 (2018).
    [https://arxiv.org/abs/1811.01430]

.. [C2015] Chambolle, Antonin, and Charles Dossal.
    On the convergence of the iterates of" FISTA.
    Journal of Optimization Theory and Applications 166.3 (2015): 25.
    [https://dl.acm.org/citation.cfm?id=2813581]

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

.. [CW2005] Combettes, P. L., and Wajs, V. R. (2005). Signal recovery by
    proximal forward-backward splitting. Multiscale Modeling & Simulation,
    4(4), 1168-1200.
    [https://pdfs.semanticscholar.org/5697/4187b4d9a8757f4d8a6fd6facc8b4ad08240.pdf]

"""

__all__ = ['cost', 'gradient', 'linear', 'algorithms', 'proximity', 'reweight']

from . import *
