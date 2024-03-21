"""
Unify functionality between Torch tensors and Numpy arrays for more readable 
code. 
"""

import torch
import numpy as np


# ------------ ##  Array and Tensor Unified Functionality  ## ----------- # 

def reshape(data, shape):
    if isinstance(data, torch.Tensor):
        return data.view(*shape)
    elif isinstance(data, np.ndarray):
        return data.reshape(*shape)
    raise ValueError(f'Data can only be a tensor or array, not {type(data)}')

def stack(data, axis):
    if isinstance(data[0], torch.Tensor):
        return torch.stack(data, dim=axis)
    elif isinstance(data[0], np.ndarray):
        return np.stack(data, axis=axis)
    raise ValueError(f'Data can only be a tensor or array, not {type(data)}')

def sum(data, axis):
    if isinstance(data, torch.Tensor):
        return data.sum(dim=axis)
    elif isinstance(data, np.ndarray):
        return data.sum(axis=tuple(axis))
    raise ValueError(f'Data can only be a tensor or array, not {type(data)}')

def to_float(data):
    if isinstance(data, torch.Tensor):
        return data.float()
    elif isinstance(data, np.ndarray):
        return data.astype(np.float32)
    raise ValueError(f'Data can only be a tensor or array, not {type(data)}')

def to_int(data):
    if isinstance(data, torch.Tensor):
        return data.int()
    elif isinstance(data, np.ndarray):
        return data.astype(np.int32)
    raise ValueError(f'Data can only be a tensor or array, not {type(data)}')

def allclose(data1, data2, rtol=1e-5, atol=1e-8, equal_nan=False):
    assert type(data1) == type(data2), f'Types: {type(data1)}, {type(data2)}'
    if isinstance(data1, torch.Tensor):
        return torch.allclose(data1, data2, rtol=rtol, atol=atol, 
                              equal_nan=equal_nan)
    elif isinstance(data1, np.ndarray):
        return np.allclose(data1, data2, rtol=rtol, atol=atol, 
                           equal_nan=equal_nan)
    raise ValueError(f'Data can only be a tensor or array, not {type(data1)}')

def any(data):
    if isinstance(data[0], torch.Tensor):
        return torch.any(data)
    elif isinstance(data[0], np.ndarray):
        return np.any(data)
    raise ValueError(f'Data can only be a tensor or array, not {type(data)}')

def nan_to_num(data, nan=0):
    if isinstance(data, torch.Tensor):
        return torch.nan_to_num(data, nan=nan)
    elif isinstance(data, np.ndarray):
        return np.nan_to_num(data, nan=nan)
    raise ValueError(f'Data can only be a tensor or array, not {type(data)}')