#!/usr/bin/env python
# encoding: utf-8

import numpy as np


def arr2hash(arr):
    return ','.join(str(i) for i in arr)


def hash2arr(hash_str):
    arr = hash_str.split(',')
    c = float(arr[0])
    x = float(arr[1])
    y = float(arr[2])
    return (c, x, y)


def get_children_pos(pos, K, S=1, D=1, P=0, filter_pad=False):
    K = (K - 1) * (D - 1) + K
    H = np.arange(S * pos[1], S * pos[1] + K, D) - P
    W = np.arange(S * pos[2], S * pos[2] + K, D) - P
    if filter_pad:
        H = H[H > 0]
        W = W[W > 0]
    res = []
    for h in H:
        for w in W:
            res.append([pos[0], h, w])
    return np.array(res)


def check_is_pad(pos, shape):
    # pos = (c,h,w)
    if pos[1] < 0 or pos[1] >= shape[0]:
        return True
    if pos[2] < 0 or pos[2] >= shape[1]:
        return True
    return False


def stop():
    import pdb
    pdb.set_trace()


def get(value, default):
    return value if value is not None else default


def feat_map_raw(activated_points, feat_map_size, raw_map_size):
    if type(feat_map_size) is int:
        feat_map_size = (feat_map_size, feat_map_size)

    if type(raw_map_size) is int:
        raw_map_size = (raw_map_size, raw_map_size)

    if type(feat_map_size) is list:
        feat_map_size = tuple(feat_map_size)
    if type(raw_map_size) is list:
        raw_map_size = tuple(raw_map_size)

    assert type(raw_map_size) is tuple and type(feat_map_size) is tuple,\
        'unsupported type of arguments'

    scale = np.array(raw_map_size).astype(
        np.float) / np.array(feat_map_size)
    return np.array(activated_points) * scale


def check_in(x, l, r):
    if x > r or x < l:
        return False
    return True
