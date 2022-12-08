import pytest


def failparam(*args, raises=ValueError):
    """Return a pytest parameterization that should raise an error."""
    return pytest.param(*args, marks=pytest.mark.raises(exception=raises))


def skipparam(*args, cond=True, reason=""):
    """Return a pytest parameterization that should raise an error."""
    return pytest.param(*args, marks=pytest.mark.skipif(cond, reason=reason))

class Dummy:
    pass
