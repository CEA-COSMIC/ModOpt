# -*- coding: utf-8 -*-

"""WRAPPERS

This module contains wrappers for adding additional features to functions

:Author: Samuel Farrens <samuel.farrens@cea.fr>

"""

from inspect import getargspec
from functools import wraps


def add_args_kwargs(func):
    """Add Args and Kwargs

    This wrapper adds support for additional arguments and keyword arguments to
    any callable function

    Parameters
    ----------
    func : function
        Callable function

    Returns
    -------
    function wrapper

    """

    @wraps(func)
    def wrapper(*args, **kwargs):

        props = getargspec(func)

        # if 'args' not in props:
        if isinstance(props[1], type(None)):

            args = args[:len(props[0])]

        if ((not isinstance(props[2], type(None))) or
                (not isinstance(props[3], type(None)))):

            return func(*args, **kwargs)

        else:

            return func(*args)

    return wrapper
