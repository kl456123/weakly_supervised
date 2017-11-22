#!/usr/bin/env python
# encoding: utf-8


def get_max_xy(data):
    if data.ndim == 0:
        return (0, 0)
    h = data.shape[0]
    w = data.shape[1]
    idx = data.ravel().argmax()
    return idx / w, idx % w
