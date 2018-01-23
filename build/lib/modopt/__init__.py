# -*- coding: utf-8 -*-

"""MODOPT PACKAGE

ModOpt is a series of Modular Optimisation tools for solving inverse problems.

:Author: Samuel Farrens <samuel.farrens@cea.fr>

"""

__all__ = ['base', 'interface', 'math', 'opt', 'plot', 'signal']

from . import *
from .base import *
from .info import __version__, __about__
