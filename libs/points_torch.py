#!/usr/bin/env python
# encoding: utf-8

import numpy as np
from libs.tools import stop, feat_map_raw


class Point(object):
    def __init__(self,
                 pos,
                 weight,
                 prior=0,
                 score=0):
        # note that if pos <0 means that it is in pad of image
        self.pos = pos
        self.weight = weight
        self.dim = 2 if np.isnan(pos[0]) else 3
        self.prior = prior
        self.score = score
        self.raw_spatial_pos = None

    def get_children_pos(self, K, S=1, D=1, P=0, filter_pad=False):
        pos = self.pos
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


class Point_2D(Point):
    def __init__(self, pos, kernel_weight, prior=0, score=0):
        assert len(kernel_weight.shape) == 1
        super(Point_2D, self).__init__(pos,
                                       kernel_weight,
                                       prior,
                                       score)
        self.dim = 2
        self.pos = np.array(self.pos).astype(np.float)
        self.pos[0] = np.nan

    # def set_score(self, point_data):
        # assert point_data

        # def get_children_pos_3D(self, shape_3D):
        # num = weight.shape[0]
        # C, H, W = shape_3D
        # assert C * H * W == num
        # return zip(np.unravel_index(np.arange(num), shape_3D))


class Point_3D(Point):
    def __init__(self, pos, weight, prior=0, score=0):
        if isinstance(weight, int) or isinstance(weight, float):
            weight = np.array(weight)
        assert len(weight.shape) == 0
        super(Point_3D, self).__init__(pos,
                                       weight,
                                       prior,
                                       score)
        self.dim = 3

    # def set_score(self, point_data):
        # assert isinstance(point_data, int) \
        # or isinstance(point_data, float),\
        # 'input data must be scar'
        # self.score = point_data * self.weight


def convert_2Dto3D(point_2D, positions_3D):
    weight = point_2D.weight
    prior = point_2D.prior
    assert len(positions_3D) == weight.size, 'positions_3D: {:d},weight.size: {:d}'.\
        format(len(positions_3D), weight.size)
    per_prior = float(prior) / weight.size
    res = []
    for idx, pos in enumerate(positions_3D):
        res.append(Point_3D(pos, weight[idx], per_prior))
    return res


def merge_points(points_3D, shape_3D):
    # only used for point_3D
    mask = np.zeros(shape_3D)
    weight = np.zeros(shape_3D)
    prior = np.zeros(shape_3D)
    for point in points_3D:
        coord = tuple(point.pos)
        # if mask[coord] ==0:
        mask[coord] = 1
        weight[coord] += point.weight
        prior[coord] += point.prior
    coord = np.nonzero(mask)
    coord = np.array(coord).T
    res = []
    for c, h, w in list(coord):
        res.append(Point_3D((c, h, w), weight[c, h, w], prior[c, h, w]))

    return res


def merge_analog_2D(points):
    pos = points[0].pos
    for point in points:
        assert (pos[1:] == point.pos[1:]).all(
        ), 'points must be in the same pos'


def merge_points_2D_better(points, shape_2D):
    feat_num = len(points[0].weight)
    mask = np.zeros(shape_2D)
    weight = np.zeros(shape_2D + (feat_num,))
    prior = np.zeros(shape_2D)
    # stop()
    for point in points:
        pos = point.pos
        coord = (int(pos[1]), int(pos[2]))
        # if mask[coord] ==0:
        mask[coord] = 1
        weight[coord] += point.weight
        prior[coord] += point.prior
    coord = np.nonzero(mask)
    coord = np.array(coord).T
    res = []
    for h, w in list(coord):
        res.append(Point_2D((np.nan, h, w), weight[h, w], prior[h, w]))

    return res


def print_point(point):
    if point.dim == 3:
        print('position:{:s}\tprior:{:.2f}\tweight:{:.2f}'.
              format(point.pos, point.prior, point.weight))
    else:
        print('position:{:s}\tprior:{:.2f}\tweight shape:{:s}'.
              format(str(point.pos),
                     point.prior,
                     str(point.weight.shape)))


def merge_points_2D(points_2D, shape_2D):
    # only used for points in the first layer
    res = []
    # feat_num = points_2D[0].weight.size
    # stop()
    # mark = np.zeros(shape_2D)
    # weight = np.zeros(shape_2D + (feat_num,))
    for point in points_2D:
        # coord = tuple(point.pos[1:].astype(np.int))
        # if mark[coord] == 0:
            # mark[coord] = 1
            # weight[coord] += point.weight
        # else:
            # stop()

        res.append(tuple(point.pos[1:]))
    return list(set(res))


# def filter_point(points_2D, scores, shape_2D):

    # pos_2D = merge_points_2D(points_2D, shape_2D)


def get_least_num(scores, ratio):
    scores = np.sort(scores)[::-1]
    sum_scores = scores.sum(axis=0)
    least_scores = sum_scores * ratio
    tmp_sum = 0
    for i, s in enumerate(scores):
        tmp_sum += s
        if tmp_sum > least_scores:
            break
    return i + 1


def filter_points_ratio(points, scores, reserve_num_ratio=1, reserve_num=5000, reserve_scores_ratio=1, max_num=50000):
    # stop()
    res = []
    num = len(points)
    reserve_num = np.minimum(int(num * reserve_num_ratio), reserve_num)
    if reserve_scores_ratio < 1:
        least_num = get_least_num(scores, reserve_scores_ratio)
    else:
        least_num = num
    reserve_num = np.maximum(reserve_num, least_num)
    reserve_num = np.minimum(max_num, reserve_num)
    sorted_idx = np.argsort(-scores)
    sorted_idx = sorted_idx[:reserve_num]
    for idx in sorted_idx:
        res.append(points[idx])

    return res, scores[sorted_idx]


def filter_negative_points(points, scores):
    keep = np.nonzero(scores > 0)
    res = []
    for idx in keep[0]:
        res.append(points[idx])
    return res, scores[keep]


def mapping_points(points, feat_size, raw_size):
    for point in points:
        pos = point.pos
        point.pos[1:] = feat_map_raw(pos[1:], feat_size, raw_size)


def threshold_system(points, scores, **kwargs):
    # stop()
    # read args
    assert len(points) == scores.size, 'each point must has its score'
    input_shape = kwargs['input_shape']
    reserve_num = kwargs['reserve_num']
    reserve_num_ratio = kwargs['reserve_num_ratio']
    reserve_scores_ratio = kwargs['reserve_scores_ratio']

    print('input number: {:d}'.format(len(points)))

    # points, scores = filter_negative_points(points, scores)
    # print 'num of positive scores of points: {:d}'.format(len(points))
    filtered_points, _ = filter_points_ratio(points,
                                             scores,
                                             reserve_num_ratio,
                                             reserve_num,
                                             reserve_scores_ratio)
    print('remain number after reserving the top scores: {:d}'.format(
        len(filtered_points)))

    filtered_points = merge_points_2D(
        filtered_points, input_shape)
    print('remain number after filtering repeat points: {:d}'.format(
        len(filtered_points)))

    return filtered_points


def post_filter_points(points, scores, **kwargs):
    assert len(points) == scores.size, 'each point must has its score'
    # stop()
    # reserve_scores_ratio = kwargs['reserve_scores_ratio']
    # reserve_num = kwargs['reserve_num']
    max_num = kwargs['max_num']
    print('post filter start to prevent from\n large amount of points')
    print('input number: {:d}'.format(len(points)))
    filtered_points, filtered_scores = filter_points_ratio(points,
                                                           scores,
                                                           max_num=max_num)
    print('remain number: {:d}'.format(len(filtered_points)))
    # points, scores = filter_negative_points(points, scores)
    # print 'num of positive scores of points: {:d}'.format(len(points))
    return filtered_points, filtered_scores


def cluster_points(points, threshold):
    # used for 2D points
    pos = points[0].pos
    num = len(points)
    pos_arr = np.zeros((num, points[0].weight.size))
    for i, point in enumerate(points):
        dim = points[0].dim
        assert dim == 2, 'points must be 2D'
        pos_arr[i] = point.weight
    norm = np.linalg.norm(pos_arr, axis=1)
    if norm[norm == 0].size > 0:
        stop()
    pos_arr = pos_arr / norm[..., np.newaxis]
    # matrix = np.dot(pos_arr, pos_arr.T)
    # matrix>threshold
    used = np.zeros((num,))
    res = []
    for i in range(num):
        if used[i]:
            continue
        used[i] = 1
        weight = pos_arr[i] * norm[i]
        prior = points[i].prior
        for j in range(num - i - 1):
            idx = i + j + 1
            if used[idx]:
                continue
            if (pos_arr[i] * pos_arr[idx]).sum(axis=0) > threshold:
                used[idx] = 1
                weight += pos_arr[idx] * norm[idx]
                prior += points[idx].prior
        res.append(Point_2D(points[i].pos, weight, prior))
    return res


# for 2D points
def helper_cluster_points(points, shape_2D, threshold=0.8):
    # build idx for points in each pos
    if points[0].dim == 3 or np.isnan(points[0].pos[2]):
        return points
    # stop()
    dict_points = {}
    h = shape_2D[0]
    w = shape_2D[1]
    for point in points:
        hash_pos = str(point.pos[1] * w + point.pos[2])
        if hash_pos not in dict_points:
            dict_points[hash_pos] = []
        dict_points[hash_pos].append(point)
    res = []
    min_num = 1
    for fixed_points in dict_points.values():
        if len(fixed_points) <= min_num:
            res += fixed_points
        else:
            res += cluster_points(fixed_points, threshold)
    return res
