# -*- coding: utf-8 -*-

"""Backend Module for GPU Compatibility.
This module contains methods for GPU Compatiblity
:Author: Chaithya G R <chaithyagr@gmail.com>
"""

import numpy as np
from importlib import util
import warnings

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
    except:
        pass


def get_array_module(data):
    """Get Array Module.
    This method returns the array module, which tells if 
    the data is residing on GPU or CPU
    Parameters
    ----------
    data : numpy.ndarray or cupy.ndarray
        Input data array
    Returns
    -------
    module
        The numpy or cupy module
    """
    if gpu_compatibility['cupy']:
        return cp.get_array_module(data)
    else:
        return np


def move_to_device(data):
    """Move data to device
    This method moves data from CPU to GPU if we have the 
    compatibility to do so. It returns the same data if 
    it is already on GPU.
    Parameters
    ----------
    data : numpy.ndarray or cupy.ndarray
        Input data array to be moved
    Returns
    -------
    cupy.ndarray
        The CuPy array residing on GPU
    """
    xp = get_array_module(data)
    if xp == cp:
        return data
    else:
        if gpu_compatibility['cupy']:
            return cp.array(data)
        else:
            warnings.warn("Cupy is not installed, cannot move data to GPU")
            return data


def move_to_cpu(data):
    """Move data to CPU
    This method moves data from GPU to CPU. 
    It returns the same data if it is already on CPU.
    Parameters
    ----------
    data : cupy.ndarray
        Input data array to be moved
    Returns
    -------
    numpy.ndarray
        The NumPy array residing on CPU
    """
    xp = get_array_module(data)
    if xp == np:
        return data
    else:
        return data.get()


def convert_to_tensor(data):
    """Convert data to a tensor
    This method converts input data to a torch tensor.
    Particularly, this method is helpful to convert CuPy array to Tensor
    Parameters
    ----------
    data : cupy.ndarray
        Input data array to be converted
    Returns
    -------
    torch.Tensor
        The tensor data
    """
    import torch
    from torch.utils.dlpack import to_dlpack, from_dlpack
    xp = get_array_module(data)
    if xp == np:
        return torch.Tensor(data)
    else:
        return from_dlpack(data.toDlpack()).float()


def convert_to_cupy_array(data):
    """Convert Tensor data to a CuPy Array
    This method converts input tensor data to a cupy array.
    Parameters
    ----------
    data : torch.Tensor
        Input Tensor to be converted
    Returns
    -------
    cupy.ndarray
        The tensor data as a CuPy array
    """
    import torch
    from torch.utils.dlpack import to_dlpack, from_dlpack
    if data.is_cuda:
        return cp.fromDlpack(to_dlpack(data))
    else:
        return data.detach().numpy()

