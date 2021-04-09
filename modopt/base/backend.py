# -*- coding: utf-8 -*-

"""BACKEND MODULE.

This module contains methods for GPU Compatiblity.

:Author: Chaithya G R <chaithyagr@gmail.com>

"""

import warnings
from importlib import util

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    import_torch = False
else:
    import_torch = True

# Handle the compatibility with variable
LIBRARIES = {
    'cupy': None,
    'tensorflow': None,
    'numpy': np,
}

if util.find_spec('cupy') is not None:
    try:
        import cupy as cp
        LIBRARIES['cupy'] = cp
    except ImportError:
        pass

if util.find_spec(tensorflow) is not None:
    try:
        from tensorflow.experimental.numpy as tnp
        LIBRARIES['tensorflow'] = tnp
    except ImportError:
        pass

from modopt.interface.errors import warn

def get_backend(backend):
    if backend not in LIBRARIES.keys() or LIBRARIES[backend] is None:
        warn(
            compute_type +
            ' backend not possible, please ensure that '
            'the optional libraries are installed. \n'
            'Reverting to numpy'
        )
        backend = 'numpy'
    return LIBRARIES['backend'], backend


def get_array_module(input_data):
    """Get Array Module.

    This method returns the array module, which tells if the data is residing
    on GPU or CPU.

    Parameters
    ----------
    input_data : numpy.ndarray or cupy.ndarray
        Input data array

    Returns
    -------
    module
        The numpy or cupy module

    """
    if LIBRARIES['tensorflow'] is not None:
        if isinstance(input_data, LIBRARIES['tensorflow'].ndarray):
            return LIBRARIES['tensorflow']
    elif LIBRARIES['cupy'] is not None:
        if isinstance(input_data, LIBRARIES['cupy'].ndarray):
            return LIBRARIES['cupy']
    else:
        return np


def change_backend(input_data, backend='cupy'):
    """Move data to device.

    This method moves data from CPU to GPU if we have the
    compatibility to do so. It returns the same data if
    it is already on GPU.

    Parameters
    ----------
    input_data : numpy.ndarray or cupy.ndarray
        Input data array to be moved

    Returns
    -------
    cupy.ndarray
        The CuPy array residing on GPU

    """
    xp = get_array_module(input_data)
    txp, target_backend = get_backend(backend)
    if xp == txp:
        return input_data
    else:
        return txp.array(input_data)


def move_to_cpu(input_data):
    """Move data to CPU.

    This method moves data from GPU to CPU.It returns the same data if it is
    already on CPU.

    Parameters
    ----------
    input_data : cupy.ndarray
        Input data array to be moved

    Returns
    -------
    numpy.ndarray
        The NumPy array residing on CPU

    """
    xp = get_array_module(input_data)

    if xp == LIBRARIES['numpy']:
        return input_data
    elif xp == LIBRARIES['cupy']:
        return input_data.get()
    elif xp == LIBRARIES['tensorflow']:
        return input_data.data.numpy()
    else:
        raise ValueError('Cant identify the kind of array!')


def convert_to_tensor(input_data):
    """Convert data to a tensor.

    This method converts input data to a torch tensor. Particularly, this
    method is helpful to convert CuPy array to Tensor.

    Parameters
    ----------
    input_data : cupy.ndarray
        Input data array to be converted

    Returns
    -------
    torch.Tensor
        The tensor data

    Raises
    ------
    ImportError
        If Torch package not found

    """
    if not import_torch:
        raise ImportError(
            'Required version of Torch package not found'
            + 'see documentation for details: https://cea-cosmic.'
            + 'github.io/ModOpt/#optional-packages',
        )

    xp = get_array_module(input_data)

    if xp == np:
        return torch.Tensor(input_data)

    return torch.utils.dlpack.from_dlpack(input_data.toDlpack()).float()


def convert_to_cupy_array(input_data):
    """Convert Tensor data to a CuPy Array.

    This method converts input tensor data to a cupy array.

    Parameters
    ----------
    input_data : torch.Tensor
        Input Tensor to be converted

    Returns
    -------
    cupy.ndarray
        The tensor data as a CuPy array

    Raises
    ------
    ImportError
        If Torch package not found

    """
    if not import_torch:
        raise ImportError(
            'Required version of Torch package not found'
            + 'see documentation for details: https://cea-cosmic.'
            + 'github.io/ModOpt/#optional-packages',
        )

    if input_data.is_cuda:
        return cp.fromDlpack(torch.utils.dlpack.to_dlpack(input_data))

    return input_data.detach().numpy()
