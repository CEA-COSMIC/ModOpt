# -*- coding: utf-8 -*-

"""MODOPT PACKAGE.

ModOpt is a series of Modular Optimisation tools for solving inverse problems.

"""

from warnings import warn

from importlib_metadata import version

from modopt.base import *

try:
    _version = version('modopt')
except Exception:  # pragma: no cover
    _version = 'Unkown'
    warn(
        'Could not extract package metadata. Make sure the package is '
        + 'correctly installed.',
    )

__version__ = _version
