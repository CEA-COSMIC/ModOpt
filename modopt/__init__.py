# -*- coding: utf-8 -*-

"""MODOPT PACKAGE

ModOpt is a series of Modular Optimisation tools for solving inverse problems.

:Author: Samuel Farrens <samuel.farrens@cea.fr>

:Version: 1.0.0

:Date: 06/12/2017

"""

__all__ = ['base', 'interface', 'math', 'plot', 'signal']

from . import *
from .base import *

version_info = (1, 0, 0)
__version__ = '.'.join(str(c) for c in version_info)
