import numpy as np
from importlib import util
import warnings

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
    if gpu_compatibility['cupy']:
        return cp.get_array_module(data)
    else:
        return np


def move_to_device(data):
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
    xp = get_array_module(data)
    if xp == np:
        return data
    else:
        return data.get()
