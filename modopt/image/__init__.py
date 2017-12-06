# -*- coding: utf-8 -*-

"""IMAGE ROUTINES

This module contains submodules for image analysis.

:Author: Samuel Farrens <samuel.farrens@gmail.com>

:Version: 1.0

:Date: 06/04/2017

References
----------

.. [C2013] Cropper et al., Defining a Weak Lensing Experiment in Space, 2013,
    MNRAS, 431, 3103C. [https://arxiv.org/abs/1210.7691]

.. [BM2007] Baker and Moallem, Iteratively weighted centroiding for
    Shack-Hartman wave-front sensors, 2007n, Optics Express, 15, 8, 5147.
    [https://www.osapublishing.org/oe/abstract.cfm?uri=oe-15-8-5147]

.. [NS2016] Ngolé and Starck, PSFs field learning based on Optimal Transport
    distances, 2016,

.. [S2005] Schneider, Weak Gravitational Lensing, 2005, arXiv.
    [https://arxiv.org/abs/astro-ph/0509252]

"""

__all__ = ['convolve', 'distort', 'quality', 'shape', 'stamp', 'stats']

from . import *
