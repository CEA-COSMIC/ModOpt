# -*- coding: utf-8 -*-

"""ERROR HANDLING ROUTINES.

This module contains methods for handing warnings and errors.

:Author: Samuel Farrens <samuel.farrens@cea.fr>

"""

import os
import sys
import warnings

try:
    from termcolor import colored
except ImportError:
    import_fail = True
else:
    import_fail = False


def warn(warn_string, log=None):
    """Warning.

    This method creates custom warning messages.

    Parameters
    ----------
    warn_string : str
        Warning message string
    log : logging.Logger, optional
        Logging structure instance (default is ``None``)

    """
    if import_fail:
        warn_txt = 'WARNING'
    else:
        warn_txt = colored('WARNING', 'yellow')

    # Print warning to stdout.
    sys.stderr.write('{0}: {1}\n'.format(warn_txt, warn_string))

    # Check if a logging structure is provided.
    if not isinstance(log, type(None)):
        warnings.warn(warn_string)


def catch_error(exception, log=None):
    """Catch error.

    This method catches errors and prints them to the terminal. It also saves
    the errors to a log if provided.

    Parameters
    ----------
    exception : str
        Exception message string
    log : logging.Logger, optional
        Logging structure instance (default is ``None``)

    """
    if import_fail:
        err_txt = 'ERROR'
    else:
        err_txt = colored('ERROR', 'red')

    # Print exception to stdout.
    stream_txt = '{0}: {1}\n'.format(err_txt, exception)
    sys.stderr.write(stream_txt)

    # Check if a logging structure is provided.
    if not isinstance(log, type(None)):
        log_txt = 'ERROR: {0}\n'.format(exception)
        log.exception(log_txt)


def file_name_error(file_name):
    """File name error.

    This method checks if the input file name is valid.

    Parameters
    ----------
    file_name : str
        File name string

    Raises
    ------
    IOError
        If file name not specified or file not found

    """
    if file_name == '' or file_name[0][0] == '-':
        raise IOError('Input file name not specified.')

    elif not os.path.isfile(file_name):
        raise IOError('Input file name {0} not found!'.format(file_name))


def is_exe(fpath):
    """Is Executable.

    Check if the input file path corresponds to an executable on the system.

    Parameters
    ----------
    fpath : str
        File path

    Returns
    -------
    bool
        True if file path exists

    """
    return os.path.isfile(fpath) and os.access(fpath, os.X_OK)


def is_executable(exe_name):
    """Check if Input is Executable.

    This method checks if the input executable exists.

    Parameters
    ----------
    exe_name : str
        Executable name

    Raises
    ------
    TypeError
        For invalid input type
    IOError
        For invalid system executable

    """
    if not isinstance(exe_name, str):
        raise TypeError('Executable name must be a string.')

    fpath, fname = os.path.split(exe_name)

    if fpath:
        res = is_exe(exe_name)

    else:
        res = any(
            is_exe(os.path.join(path, exe_name))
            for path in os.environ['PATH'].split(os.pathsep)
        )

    if not res:
        message = (
            '{0} does not appear to be a valid executable on this system.'
        )
        raise IOError(message.format(exe_name))
