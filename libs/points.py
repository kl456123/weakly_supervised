#!/usr/bin/env python
# encoding: utf-8

import numpy as np


class Point(object):
    def __init__(self, pos, weight, prior=0):
        # note that if pos <0 means that it is in pad of image
        self.pos = pos
        self.weight = weight
        self.dim = 2 if np.isnan(pos[0]) else 3
        self.prior = prior
        self.score = 0

    def get_children_pos(self, K, S=1, D=1, P=0):
        pos = self.pos
        K = (K - 1) * (D - 1) + K
        H = np.arange(S * pos[1], S * pos[1] + K, D) - P
        W = np.arange(S * pos[2], S * pos[2] + K, D) - P
        res = []
        for h in H:
            for w in W:
                res.append([np.nan, h, w])
        return np.array(res)


class Point_2D(Point):
    def __init__(self, pos, kernel_weight, prior=0):
        assert len(kernel_weight.shape) == 1
        super(Point_2D, self).__init__(pos, kernel_weight, prior)

    # def set_score(self, point_data):
        # assert point_data

        # def get_children_pos_3D(self, shape_3D):
        # num = weight.shape[0]
        # C, H, W = shape_3D
        # assert C * H * W == num
        # return zip(np.unravel_index(np.arange(num), shape_3D))


class Point_3D(Point):
    def __init__(self, pos, weight, prior=0):
        if isinstance(weight, int) or isinstance(weight, float):
            weight = np.array(weight)
        assert len(weight.shape) == 0
        super(Point_3D, self).__init__(pos, weight, prior)

    # def set_score(self, point_data):
        # assert isinstance(point_data, int) \
        # or isinstance(point_data, float),\
        # 'input data must be scar'
        # self.score = point_data * self.weight


def convert_2Dto3D(point_2D, positions_3D):
    weight = point_2D.weight
    prior = point_2D.prior
    assert len(positions_3D) == weight.size
    per_prior = float(prior) / weight.size
    res = []
    for idx, pos in enumerate(positions_3D):
        res.append(Point_3D(pos, weight[idx], per_prior))
    return res


def merge_points(points_3D, shape_3D):
    # only used for point_3D
    mask = np.zeros(shape_3D)
    score = np.zeros(shape_3D)
    prior = np.zeros(shape_3D)
    for point in points_3D:
        # coord = tuple(point.pos)
        # if mask[coord] ==0:
        mask[coord] = 1
        score[coord] += point.weight
        prior[coord] += point.prior
    coord = np.nonzero(mask)
    res = []
    for c, h, w in coord:
        res.append(Point_3D((c, h, w), score[c, h, w], prior[c, h, w]))

    return res


def print_point(point):
    if point.dim == 3:
        print 'position:{:s}\tprior:{:.2f}\tweight:{:.2f}'.\
            format(point.pos, point.prior, point.weight)
    else:
        print 'position:{:s}\tprior:{:.2f}\tweight shape:{:s}'.\
            format(str(point.pos),
                   point.prior,
                   str(point.weight.shape))


def merge_points_2D(points_2D, shape_2D):
    # only used for points in the first layer
    res = []
    for point in points_2D:
        res.append(tuple(point.pos[1:]))
    return list(set(res))
