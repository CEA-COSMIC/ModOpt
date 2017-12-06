# -*- coding: utf-8 -*-

"""ERROR HANDLING ROUTINES

This module contains methods for handing warnings and errors.

:Author: Samuel Farrens <samuel.farrens@gmail.com>

:Version: 1.2

:Date: 23/10/2017

"""


import sys
import os.path
import warnings
try:
    from termcolor import colored
except ImportError:
    import_fail = True
else:
    import_fail = False


def warn(warn_string, log=None):
    """Warning

    This method creates custom warning messages.

    Parameters
    ----------
    warn_string : str
        Warning message string
    log : instance, optional
        Logging structure instance

    """

    if import_fail:
        warn_txt = 'WARNING'
    else:
        warn_txt = colored('WARNING', 'yellow')

    # Print warning to stdout.
    sys.stderr.write(warn_txt + ': ' + warn_string + '\n')

    # Check if a logging structure is provided.
    if not isinstance(log, type(None)):
        warnings.warn(warn_string)


def catch_error(exception, log=None):
    """Catch error

    This method catches errors and prints them to the terminal. It also saves
    the errors to a log if provided.

    Parameters
    ----------
    exception : str
        Exception message string
    log : instance, optional
        Logging structure instance

    """

    if import_fail:
        err_txt = 'ERROR'
    else:
        err_txt = colored('ERROR', 'red')

    # Print exception to stdout.
    stream_txt = err_txt + ': ' + str(exception) + '\n'
    sys.stderr.write(stream_txt)

    # Check if a logging structure is provided.
    if not isinstance(log, type(None)):
        log_txt = 'ERROR: ' + str(exception) + '\n'
        log.exception(log_txt)


def file_name_error(file_name):
    """File name error

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
        raise IOError('Input file name [%s] not found!' % file_name)
