#!/usr/bin/env python
# encoding: utf-8


import torch as T
import numpy as np


def tensor(arr, dtype=T.cuda.FloatTensor):
    if isinstance(arr, list) or isinstance(arr, tuple):
        if T.is_tensor(arr[0]):
            return T.stack(arr)
        else:
            arr = np.array(arr)
    if isinstance(arr, np.ndarray):
        return T.from_numpy(arr).type(dtype)
    if T.is_tensor(arr):
        return arr


def clear():
    T.cuda.empty_cache()
