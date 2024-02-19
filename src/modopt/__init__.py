"""MODOPT PACKAGE.

ModOpt is a series of Modular Optimisation tools for solving inverse problems.

"""

from warnings import warn

from importlib_metadata import version

from modopt.base import np_adjust, transform, types, wrappers, observable

__all__ = ["np_adjust", "transform", "types", "wrappers", "observable"]

try:
    _version = version("modopt")
except Exception:  # pragma: no cover
    _version = "Unkown"
    warn(
        "Could not extract package metadata. Make sure the package is "
        + "correctly installed.",
    )

__version__ = _version
