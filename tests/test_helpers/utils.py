"""
Some helper functions for the test parametrization.

They should be used inside ``@pytest.mark.parametrize`` call.

:Author: Pierre-Antoine Comby <pierre-antoine.comby@cea.fr>
"""

import pytest


def failparam(*args, raises=None):
    """Return a pytest parameterization that should raise an error."""
    if not issubclass(raises, Exception):
        raise ValueError("raises should be an expected Exception.")
    return pytest.param(*args, marks=[pytest.mark.xfail(exception=raises)])


def skipparam(*args, cond=True, reason=""):
    """Return a pytest parameterization that should be skip if cond is valid."""
    return pytest.param(*args, marks=[pytest.mark.skipif(cond, reason=reason)])


class Dummy:
    """Dummy Class."""

    pass
