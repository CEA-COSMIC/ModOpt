# -*- coding: utf-8 -*-

"""LOGGING ROUTINES

This module contains methods for handing logging.

:Author: Samuel Farrens <samuel.farrens@cea.fr>

"""

from __future__ import print_function
import sys
import logging


def set_up_log(filename, verbose=True):
    """Set up log

    This method sets up a basic log.

    Parameters
    ----------
    filename : str
        Log file name

    Returns
    -------
    logging.Logger instance

    """

    # Add file extension.
    filename += '.log'

    if verbose:
        print('Preparing log file:', filename)

    # Capture warnings.
    logging.captureWarnings(True)

    # Set output format.
    formatter = logging.Formatter(fmt='%(asctime)s %(message)s',
                                  datefmt='%d/%m/%Y %H:%M:%S')

    # Create file handler.
    fh = logging.FileHandler(filename=filename, mode='w')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # Create log.
    log = logging.getLogger(filename)
    log.setLevel(logging.DEBUG)
    log.addHandler(fh)

    # Send opening message.
    log.info('The log file has been set-up.')

    return log


def close_log(log, verbose=True):
    """Close log

    This method closes and active logging.Logger instance.

    Parameters
    ----------
    log : logging.Logger
        Logging instance

    """

    if verbose:
        print('Closing log file:', log.name)

    # Send closing message.
    log.info('The log file has been closed.')

    # Remove all handlers from log.
    [log.removeHandler(handler) for handler in log.handlers]
