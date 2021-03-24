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
gpu_compatibility = {
    'cupy': False,
    'cupy-cudnn': False,
}

if util.find_spec('cupy') is not None:
    try:
        import cupy as cp
        gpu_compatibility['cupy'] = True

        if util.find_spec('cupy.cuda.cudnn') is not None:
            gpu_compatibility['cupy-cudnn'] = True
    except ImportError:
        pass


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
    if gpu_compatibility['cupy']:
        return cp.get_array_module(input_data)

    return np


def move_to_device(input_data):
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

    if xp == cp:
        return input_data

    if gpu_compatibility['cupy']:
        return cp.array(input_data)

    warnings.warn('Cupy is not installed, cannot move data to GPU')

    return input_data


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

    if xp == np:
        return input_data

    return input_data.get()


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
